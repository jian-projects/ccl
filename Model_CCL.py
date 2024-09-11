
import torch, os, random, copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict

from transformers import logging
logging.set_verbosity_error()

from utils_curriculum import CustomSampler
from utils_contrast import scl
from utils_processor import *
from utils_model import *

# config 中已经添加路径了
from data_loader import ALSCDataModule


class ALSCDataset_(ALSCDataModule):
    def setup_(self, tokenizer, stage=None):
        self.tokenizer = tokenizer
        for stage, samples in self.datas.items():
            if samples is None: continue
            self.info['class_category'][stage] = {l: 0 for l in self.tokenizer_['labels']['i2l'].keys()}
            for sample in samples:
                if 'sentence' not in sample: sample['sentence'] = ' '.join(sample['tokens'])
                embedding_sent = tokenizer.encode(sample['sentence'], return_tensors='pt')[0]
                embedding_asp = tokenizer.encode(sample['aspect'], return_tensors='pt')[0][1:]
                sample['input_ids'] = torch.cat([embedding_sent, embedding_asp])
                sample['attention_mask'] = torch.ones_like(sample['input_ids'])
                sample['token_type_ids'] = torch.cat([torch.zeros_like(embedding_sent), torch.ones_like(embedding_asp)])
                sample['label'] = self.tokenizer_['labels']['l2i'][sample['polarity']]
                sample['stage'] = stage
                
                self.info['class_category'][stage][sample['label']] += 1
    
    def setup(self, samples, stage=None):
        tokenizer = self.tokenizer
        for s in samples:
            aspect, tokens = copy.deepcopy(s['aspect']), copy.deepcopy(s['tokens'])
            aspect_noise = []
            for c in aspect:
                if c == ' ': 
                    aspect_noise.append(c)
                    continue
                if random.random() < 0.05: 
                    if random.random() < 0.5: aspect_noise.extend([c, c])
                else: aspect_noise.append(c)

            aspect = ''.join(aspect_noise)
            aspect_token = aspect.split(' ')
            assert len(aspect_token) == s['aspect_pos'][1]-s['aspect_pos'][0]
            tokens[s['aspect_pos'][0]:s['aspect_pos'][1]] = aspect_token
            sentence = ' '.join(tokens)

            embedding_sent = tokenizer.encode(sentence, return_tensors='pt')[0]
            embedding_asp = tokenizer.encode(aspect, return_tensors='pt')[0][1:]
            s['input_ids'] = torch.cat([embedding_sent, embedding_asp])
            s['attention_mask'] = torch.ones_like(s['input_ids'])
            s['token_type_ids'] = torch.cat([torch.zeros_like(embedding_sent), torch.ones_like(embedding_asp)])
            s['label'] = self.tokenizer_['labels']['l2i'][s['polarity']]
                    
        return samples
    
    def collate_fn(self, samples):
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs


def config_for_model(args, scale='base'):
    scale = args.model['scale'] if 'scale' in args.model else scale
    if args.model['arch'] == 'bert':
        if scale == 'base':
            args.model['plm'] = args.file['plm_dir'] + f'bert-{scale}-uncased'
        else: args.model['plm'] = args.file['plm_dir'] + f'bert-{scale}'
    if args.model['arch'] == 'deberta':
        args.model['plm'] = args.file['plm_dir'] + f'deberta-{scale}'
    if args.model['arch'] == 'roberta':
        args.model['plm'] = args.file['plm_dir'] + f'roberta-{scale}'

    sbert = "all-roberta-large-v1" if scale == 'large' else 'all-distilroberta-v1'
    args.model['sbert'] = f"{args.file['plm_dir']}/sbert/{sbert}" # 
    args.model['data'] = args.file['cache_dir']+f"{args.model['name']}_for_all.pt"
    args.model['optim_sched'] = ['AdamW', 'linear']
    args.model['store_first'] = True
    return args

def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    
    ## 2. 导入数据
    data_path = args.model['data']
    if os.path.exists(data_path):
        dataset = torch.load(data_path)
    else:
        data_dir = f"{args.file['data_dir']}{args.train['tasks'][-1]}/"
        dataset = ALSCDataset_(data_dir,  args.train['batch_size'], num_workers=0)   
        torch.save(dataset, data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
    dataset.tokenizer = tokenizer
    dataset.setup_(tokenizer)
    dataset.batch_cols = {
        'index': -1,
        'input_ids': dataset.tokenizer.pad_token_id,
        'attention_mask': 0,
        'token_type_ids': 0, 
        'label': -1,
    }

    ## 3. 导入模型
    model = CCL(
        args=args,
        dataset=dataset,
    )
    return model, dataset
   

class CCL(ModelForClassification):
    def __init__(self, args, dataset, plm=None):
        super().__init__() 
        self.args = args
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.aspect_count(dataset.datas['train'])

        if args.model['use_adapter']:
            from utils_adapter import auto_load_adapter
            self.plm_model = auto_load_adapter(args, plm=plm if plm is not None else args.model['plm'])
        else: self.plm_model = AutoModel.from_pretrained(plm if plm is not None else args.model['plm'])
        self.plm_pooler = PoolerAll(self.plm_model.config)    
        self.hidden_size = self.plm_model.config.hidden_size

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.loss_ce = nn.CrossEntropyLoss() 

        self.bank = {
            'label': torch.tensor([s['label'] for s in dataset.datas['train']]),
            'aspect': [None for s in dataset.datas['train']],
            'sentence': [None for s in dataset.datas['train']],
        }
        self.use_scl = args.model['use_scl']
        self.weight = args.model['weight']

    def aspect_count(self, trainset):
        aspect_term = [sample['aspect'] for sample in trainset]
        aspect_dict = defaultdict(list)
        for i, asp in enumerate(aspect_term):
            aspect_dict[asp].append(i)
        self.aspect_index_dict = aspect_dict
        self.aspect_num_dict = {asp: len(index) for asp, index in aspect_dict.items()}
        self.aspect_num_list = [len(index) for asp, index in aspect_dict.items()]

    def get_indices(self, dataset):
        if None in self.bank['aspect']: return None

        aspect_vec = torch.stack(self.bank['aspect']).to(self.plm_model.device)
        aspect_sim_list, indices, indices_ = [], [], list(range(len(aspect_vec)))
        for i in range(0, aspect_vec.size(0), 128):
            batch = aspect_vec[i:i + 128]
            batch_sim = F.cosine_similarity(batch.unsqueeze(1), aspect_vec.unsqueeze(0), dim=-1)
            aspect_sim_list.append(batch_sim)
        aspect_sim = torch.cat(aspect_sim_list, dim=0)

        while indices_:
            first = random.choice(indices_) # 随机选择一个
            sim = aspect_sim[first]
            sim[indices] = -1e8 # 不能重复选择
            sim_sorted = sim.argsort(descending=True).tolist()

            batch_idx = sim_sorted[:self.args.train['batch_size']]
            for idx in batch_idx: 
                if len(indices_) == 0: break
                indices_.remove(idx)
                indices.append(idx)
                
        return indices

    def epoch_deal(self, epoch):
        indices = self.get_indices(self.dataset)
        # 课程顺序 (indices 是batch的list)
        if indices is not None:
            self.dataset.loader['train'] = DataLoader(
                self.dataset.datas['train'], 
                batch_size=self.args.train['batch_size'], 
                sampler=CustomSampler(indices),
                num_workers=0,
                collate_fn=self.dataset.collate_fn,
                )

    def encode(self, inputs, methods=['cls', 'asp', 'all']):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        plm_out = self.plm_model(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
            )

        hidden_states = self.plm_pooler(plm_out.last_hidden_state)
        #hidden_states = plm_out.last_hidden_state

        outputs = {}
        for method in methods:
            if method == 'cls': outputs['cls'] = hidden_states[:,0]
            if method == 'all': outputs['all'] = hidden_states
            if method == 'asp':
                token_type_ids = inputs['token_type_ids'] # token_type_ids 的平均值
                outputs['asp'] = torch.stack([torch.mean(hidden_states[bi][tmp.bool()], dim=0) for bi, tmp in enumerate(token_type_ids)])

        return outputs

    def forward(self, inputs, stage='train'):
        outputs = self.encode(inputs, methods=['cls', 'asp'])
        logits = self.classifier(outputs['cls'])
        loss = self.loss_ce(logits, inputs['label'])

        if stage == 'train':
            self.store_asp_features(inputs, outputs) # 存储 asp 表示
        
        if stage == 'train' and self.use_scl > 0:
            loss_scl = scl(outputs['cls'], inputs['label'], temp=1) # 让相似的 asp 更相似
            loss = loss*(1-self.weight) + loss_scl*self.weight

        return {
            'loss':   loss,
            'logits': logits,
            'preds':  torch.argmax(logits, dim=-1).cpu(),
            'labels': inputs['label'],
        }
    
    def store_asp_features(self, inputs, outputs):
        for i, idx in enumerate(inputs['index']):         
            self.bank['aspect'][idx] = outputs['asp'][i].detach().cpu()

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.use_scl += 1

        output = self(batch, stage='train')
        self.training_step_outputs.append(output)

        return {
            'loss': output['loss']
        }


def scl(embeddings, labels, temp=0.3):
    """
    calculate the contrastive loss (optimized)
    embedding: [bz, dim]
    label: [bz, bz] # True or False
    """
    # cosine similarity between embeddings
    cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1) / temp
    # cosine_sim = torch.matmul(embeddings, embeddings.T) / temp
    # remove diagonal elements from matrix
    mask = ~torch.eye(cosine_sim.shape[0], dtype=bool,device=cosine_sim.device)
    dis = cosine_sim[mask].reshape(cosine_sim.shape[0], -1)
    # apply exp to elements
    dis_exp = torch.exp(dis)
    row_sum = dis_exp.sum(dim=1) # calculate row sum

    # calculate contrastive loss
    contrastive_loss, contrastive_num = 0, 0
    for i, exp in enumerate(dis_exp):
        mark = torch.tensor([l==labels[i] for i_,l in enumerate(labels) if i_!=i])
        if mark.sum() == 0: continue
        inner_sum = torch.log(exp[mark] / row_sum[i]).sum()
        contrastive_loss += inner_sum / (-mark.sum()) 
        contrastive_num += 1

    return contrastive_loss/contrastive_num if contrastive_num else 0
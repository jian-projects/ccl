import os, torch, json, copy
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score


class Metrics(object):
    def __init__(self, base_metric='f1', dataset=None) -> None:
        self.base = base_metric
        self.results = {
            'train': { self.base: 0, 'loss': 0 }, 
            'valid': { self.base: 0 }, 
            'test':  { self.base: 0 }
            }
        
        self.dataset = dataset # 可有可无

    def _score(self, results, stage='train'):
        preds = np.concatenate([rec['preds'].cpu().numpy() for rec in results])
        truthes = np.concatenate([rec['labels'].cpu().numpy() for rec in results])
        losses = [rec['loss'].item() for rec in results]

        score_f1 = round(f1_score(truthes, preds, average='weighted'), 4)
        score_acc = round(accuracy_score(truthes, preds), 4)
        score_loss = round(sum(losses)/len(losses), 3)

        return {
            'f1'  : score_f1,
            'acc' : score_acc,
            'loss': score_loss
        }


class ALSCDataModule(Dataset):
    def __init__(self, data_dir, batch_size=2, num_workers=8) -> None:
        super().__init__()
        self.name = ['alsc', data_dir.split('/')[-2]]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_init() # dataset initialize
        self.prepare_data(stages=['train', 'test']) # 
        self.get_tokenizer_(self.datas['train'], names=['polarity'])
        self.num_classes = len(self.tokenizer_['labels']['l2i'])
        self.metrics = Metrics(base_metric='f1')

        self.datas['valid'] = self.datas['test'] # no validation set

    def dataset_init(self):
        self.info = {
            'max_seq_token_num': {}, # 句子 最长长度
            'max_asp_token_num': {}, # aspect 最长长度
            'total_samples_num': {}, # 样本数量
            'class_category': {},    # 类别统计
        }

        # 初始化数据集要保存的内容 
        self.datas, self.loader = {}, {}
        self.tokenizer_ = {
            'labels': { 'l2i': {}, 'i2l': {}, 'count': {} }
        }

    def get_tokenizer_(self, samples, names):
        if not isinstance(names, list): names = [names]
        if 'polarity' in names:
            for samp in samples:
                value = samp['polarity']
                if value not in self.tokenizer_['labels']['l2i']:
                    self.tokenizer_['labels']['l2i'][value] = len(self.tokenizer_['labels']['l2i'])
                    self.tokenizer_['labels']['i2l'][len(self.tokenizer_['labels']['i2l'])] = value
                    self.tokenizer_['labels']['count'][value] = 1
                self.tokenizer_['labels']['count'][value] += 1

    def prepare_data(self, stages=['train', 'valid', 'test']):
        for stage in stages:
            raw_path = f'{self.data_dir}/{stage}.multiple.json'
            if not os.path.exists(raw_path): return None
            with open(raw_path, 'r', encoding='utf-8') as fp: raw_samples, samples = json.load(fp), []

            for sample in tqdm(raw_samples):
                aspects = sample['aspects']
                for aspect in aspects:
                    temp = copy.deepcopy(sample)
                    temp['index'] = len(samples)
                    temp['aspect'] = ' '.join(aspect['term'])
                    temp['aspect_pos'] = [aspect['from'], aspect['to']]
                    if 'tokens' not in temp: temp['tokens'] = temp['token']
                    if 'sentence' not in temp: temp['sentence'] = ' '.join(temp['tokens'])
                    if ' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]]) != temp['aspect']:
                        print(f"{' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]])} -> {temp['aspect']}")
                    temp['polarity'] = aspect['polarity']
                    samples.append(temp)

            self.datas[stage] = samples

    def setup(self, tokenizer, stage=None):
        self.tokenizer = tokenizer
        for stage, samples in self.datas.items():
            if samples is None: continue
            self.info['class_category'][stage] = {l: 0 for l in self.tokenizer_['labels']['i2l'].keys()}
            for sample in samples:
                # asp_pos = sample['aspect_pos']
                # sentence = ' '.join(sample['tokens'][0:asp_pos[0]]) + ' [' + sample['aspect'] + '] ' + ' '.join(sample['tokens'][asp_pos[1]:])
                # aspect = '[' + sample['aspect'] + ']'
                # embedding = tokenizer.encode_plus(sentence, aspect, return_tensors='pt')

                if 'sentence' not in sample: sample['sentence'] = ' '.join(sample['tokens'])
                embedding = tokenizer.encode_plus(sample['sentence'], sample['aspect'], return_tensors='pt')
                sample['input_ids'] = embedding['input_ids'].squeeze(dim=0)
                sample['attention_mask'] = embedding['attention_mask'].squeeze(dim=0)
                # sample['token_type_ids'] = embedding['token_type_ids'].squeeze(dim=0)
                sample['label'] = self.tokenizer_['labels']['l2i'][sample['polarity']]
                
                self.info['class_category'][stage][sample['label']] += 1

    def get_dataloader(self, batch_size=None):
        if batch_size: self.batch_size = batch_size
        for stage, _ in self.datas.items():
            if stage=='train': self.loader[stage] = self.train_dataloader()
            if stage=='valid': self.loader[stage] = self.val_dataloader()
            if stage=='test':  self.loader[stage] = self.test_dataloader()
        return self.loader

    def train_dataloader(self):
        return DataLoader(
            self.datas['train'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.datas['valid'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datas['test'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    
    def collate_fn(self, samples):
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs
    

if __name__ == '__main__':
    data_dir = '/home/jzq/My_Codes/CodeFrame/Datasets/Textual/absa/twi/'
    dataset = ALSCDataModule(data_dir)
    plm_dir = None
    tokenizer = AutoTokenizer.from_pretrained(plm_dir)
    dataset.setup(tokenizer)
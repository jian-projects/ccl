from torch.utils.data import DataLoader, Sampler


class CustomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        # return (i for i in self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# # 使用自定义的索引列表创建 CustomSampler
# indices = [7, 2, 1, 6, 4, 3, 5, 0]  # 自定义的索引列表
# sampler = CustomSampler(indices)
# data_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
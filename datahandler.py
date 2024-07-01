import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class CustomDataset(Dataset):
    def __init__(self, x, y):
        super(CustomDataset, self).__init__()

        # 对 x 和 y 进行归一化
        self.x = (x - np.mean(x)) / np.std(x)
        self.y = (y - np.mean(y)) / np.std(y)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
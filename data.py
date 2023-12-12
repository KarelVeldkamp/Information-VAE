import torch
from torch.utils.data import DataLoader, Dataset

class SimDataset(Dataset):
    """
    Torch dataset for item response data in numpy array
    """
    def __init__(self, X, device='cpu'):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        self.x_train = torch.tensor(X, dtype=torch.float32)
        missing = torch.isnan(self.x_train)
        self.x_train[missing] = 0
        self.mask = (~missing).int()
        self.x_train = self.x_train.to(device)
        self.mask = self.mask.to(device)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.mask[idx]
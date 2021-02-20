from abc import ABC

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """
    Implementation of Dataset,

    It takes two images from style (behance dataset) and content (CelebA dataset),
    and returns a tuple with them.
    """

    def __init__(self):
        super(CustomDataset, self).__init__()
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class CustomDataLoader(pl.LightningDataModule, ABC):
    def __init__(self):
        super(CustomDataLoader, self).__init__()
        pass

    def prepare_data(self):
        """
        Checking if data is unarchived or

        archives with images exists and need to be unarchived.
        :return: None
        """
        pass

    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

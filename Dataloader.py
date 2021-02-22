import os
import zipfile
from abc import ABC
from typing import List, Union, Optional

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
    def __init__(self,
                 content_train_names: List[str],
                 style_train_names: List[str],
                 content_val_names: Union[List[str], None],
                 style_val_names: Union[List[str], None],
                 dataset_config
                 ):
        """

        :param content_train_names: name of training content datasets
        :param style_train_names: name of training style datasets
        :param content_val_names: name of validation content datasets (None if use datasets from training)
        :param style_val_names: name of testing style datasets (None if use datasets from training)
        :param dataset_config: config where paths to archives, folders and sizes are stored.
        """
        super(CustomDataLoader, self).__init__()

        self.content_train_names = content_train_names
        self.style_train_names = style_train_names

        assert not content_train_names, "[] is passed as training content dataset"
        assert not style_train_names, "[] is passed as training style dataset"

        self.content_val_names = content_val_names or content_train_names
        self.style_val_names = style_val_names or style_train_names

        self.train_dataset = None
        self.val_dataset = None

        self.dataset_config = dataset_config

    def prepare_data(self):
        """
        Checking if data is unarchived or archives with images exists and need to be unarchived.

        :return: None
        """

        names = [self.content_train_names, self.style_train_names, self.content_val_names, self.style_val_names]

        # TODO(create needed folders content/styles)

        for name in names:
            for dataset in name:
                # Reading configs form dataset files
                dataset_archive = self.dataset_config[dataset + "_archive"]
                dataset_folder = self.dataset_config[dataset + "_path"]

                if os.path.isdir(dataset_folder):
                    # Folder exists, skipping unpacking
                    print(f"Dataset folder: {dataset_folder} exists, skipping unzipping...")
                    continue

                else:
                    if os.path.isfile(dataset_archive):
                        # Archive exists, unpacking data

                        print(f"Unzipping {dataset_archive} into {dataset_folder}...")
                        with zipfile.ZipFile(dataset_archive, 'r') as zip_ref:
                            zip_ref.extractall(dataset_folder)

                        print("Done!")

                    else:
                        raise FileNotFoundError(f"Archive {dataset_archive} does not exists!")

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

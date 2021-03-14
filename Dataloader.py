import os
import zipfile
from abc import ABC
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.utils.data.dataset import ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    """
    Implementation of Dataset, wrapper on ImageDataset,
    concat different dataset and allows setting the size of their dataset sizes explicitly.
    """

    def __init__(self, dataset_names, dataset_configs, transform):
        super(CustomDataset, self).__init__()

        self.dataset = []

        for name in dataset_names:

            _dataset = ImageFolder(root=dataset_configs[name + '_path'], transform=transform)

            _dataset_size = dataset_configs[name + "_size"]
            assert type(_dataset_size) == int, "Dataset size should be int number!"

            if _dataset_size != -1:
                _dataset, _ = random_split(_dataset, [_dataset_size, len(_dataset) - _dataset_size])

            self.dataset.append(_dataset)

        self.dataset = ConcatDataset(self.dataset)

    def __getitem__(self, item):
        image = self.dataset[item]

        return image

    def __len__(self):
        return len(self.dataset)


class MergeDatasets(Dataset):
    """
    Unites two datasets, shrinks them by minimum length of datasets
    """

    def __init__(self, first_dataset, second_dataset):
        super(MergeDatasets).__init__()

        min_length = min(len(first_dataset), len(second_dataset))

        self.first_dataset_cut = random_split(first_dataset, [min_length, len(first_dataset) - min_length])
        self.second_dataset_cut = random_split(second_dataset, [min_length, len(second_dataset) - min_length])

        assert len(self.first_dataset_cut) == len(self.second_dataset_cut), "Защита от дурачка, ы"
        print(f"Length of dataset: {min_length}")

    def __getitem__(self, index):
        image1 = self.first_dataset_cut[index]
        image2 = self.second_dataset_cut[index]

        return image1, image2

    def __len__(self):
        return len(self.first_dataset_cut)


class CustomDataLoader(pl.LightningDataModule, ABC):
    def __init__(self,
                 content_train_names: List[str],
                 style_train_names: List[str],
                 dataset_config
                 ):
        """
        Implementation of Dataloader, now only support train_dataloader.

        :param content_train_names: name of training content datasets
        :param style_train_names: name of training style datasets
        :param dataset_config: config where paths to archives, folders and sizes are stored.
        """
        super(CustomDataLoader, self).__init__()

        self.content_train_names = content_train_names
        self.style_train_names = style_train_names

        assert not content_train_names, "[] is passed as training content dataset"
        assert not style_train_names, "[] is passed as training style dataset"

        self.train_dataset = None
        self.val_dataset = None

        self.dataset_config = dataset_config

        self.custom_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.dataset_config['image_size']),
            transforms.ToPILImage(),

            # VGG works better using this coef.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        """
        Checking if data is unarchived or archives with images exists and need to be unarchived.
        """

        names = [self.content_train_names, self.style_train_names]

        # Checking that folders exists
        if not os.path.isdir('data'):
            os.mkdir('data')

        if not os.path.isdir('data/content'):
            os.mkdir('data/content')

        if not os.path.isdir('data/style'):
            os.mkdir('data/style')

        # Checking folders and unpacking archives
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
        """
        Here I merge different datasets into train and val dataset
        :param stage: Dunno know what it is
        :return:
        """
        content_train_dataset = CustomDataset(
            dataset_names=self.content_train_names,
            transform=self.custom_transforms,
            dataset_configs=self.dataset_config
        )

        style_train_dataset = CustomDataset(
            dataset_names=self.style_train_names,
            transform=self.custom_transforms,
            dataset_configs=self.dataset_config
        )

        print(f"Length of content training dataset: {len(content_train_dataset)}")
        print(f"Length of style training dataset: {len(style_train_dataset)}")

        union_dataset = MergeDatasets(first_dataset=content_train_dataset, second_dataset=style_train_dataset)

        train_images = int((1 - self.dataset_config['val_percentage']) * len(union_dataset))
        self.train_dataset, self.val_dataset = random_split(
            union_dataset, [train_images, len(union_dataset) - train_images]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.dataset_config['batch_size'],
            num_workers=self.dataset_config['num_workers'],
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.dataset_config['batch_size'],
            num_workers=self.dataset_config['num_workers'],
            drop_last=True,
            shuffle=False
        )

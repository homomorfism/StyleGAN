# This is training script

import argparse

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from Dataloader import CustomDataLoader
from models.StyleGAN import StyleGAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', default='configs/dataset.yaml')
    parser.add_argument('--model_config', default='configs/model.yaml')
    args = parser.parse_args()

    with open(args.model_config) as model_config, open(args.dataset_config) as dataset_config:
        model_file = yaml.safe_load(model_config)
        dataset_file = yaml.safe_load(dataset_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_file['checkpoints_folder'],
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='gen_loss',
        mode='min'
    )

    logger = TensorBoardLogger(
        save_dir=model_file['tensorboard_log_dir'],
        name='StyleGAN'
    )

    dataloader = CustomDataLoader(
        content_train_names=dataset_file['content_train_names'],
        style_train_names=dataset_file['style_train_names'],
        dataset_config=dataset_config
    )

    train_loader = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()

    model = StyleGAN(
        dataset_configs=dataset_file,
        model_configs=model_file
    )

    print("Starting training...")

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=model_file['accumulate_grad_batches']
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()

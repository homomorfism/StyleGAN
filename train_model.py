import argparse
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# This is training script


def main():
    # We can set dataset and resume weights when starting training
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', 'configs/dataset.yaml')
    parser.add_argument('--resume', default="last.ckpt")

    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    checkpoint_callback = ModelCheckpoint(
        filepath=config['checkpoints_folder'],
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='generator_loss',
        mode='min'
    )

    logger = TensorBoardLogger(
        save_dir=config['tensorboard_log_dir'],
        name='StyleGAN'
    )

    # Datasets ....
    # Dataloader ...

    if args.resume:
        print("Resuming from latest checkpoint ...")

        trainer = pl.Trainer(
            gpus=1,
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=os.path.join(config['checkpoint_folder'], args.resume),
            accumulate_grad_batches=config['accumulate_grad_batches'],
            fast_dev_run=True,
        )
    else:
        print("Starting training ...")

        trainer = pl.Trainer(
            gpus=1,
            logger=logger,
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=config['accumulate_grad_batches'],
            fast_dev_run=True,
        )

    # trainer.fit()


if __name__ == '__main__':
    main()

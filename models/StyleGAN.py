import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models.Discriminator import Discriminator
from models.Generator import Generator


class StyleGAN(pl.LightningModule):
    """
    Implementation of StyleGAM.

    Reading ingo about datasets and model parameters from config files.
    """

    def __init__(self, dataset_configs, model_configs, ):
        super(StyleGAN, self).__init__()

        self.dataset_configs = dataset_configs
        self.model_configs = model_configs
        # Do we need to store logger for passing there images & losses ?

        self.generator = Generator()
        self.discriminator = Discriminator(
            relu_negative_slope=model_configs['relu_negative_slope'],
            style_classes=dataset_configs['style_classes']
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass

    def generator_loss(self):
        pass

    def discriminator_loss(self):
        pass

    def generator_step(self):
        pass

    def discriminator_step(self):
        pass

    def configure_optimizers(self):
        lr = self.model_configs['lr_start']
        betas = (self.model_configs['adam_b1'], self.model_configs['adam_b2'])
        step_size = self.model_configs['lr_step_size']

        gen_opt = Adam(self.generator.parameters(), lr=lr, betas=betas)
        discr_opt = Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        gen_lr_sched = StepLR(gen_opt, step_size=step_size)

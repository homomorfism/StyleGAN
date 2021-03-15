import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid

from models.Discriminator import Discriminator
from models.Generator import Generator
from models.utils import DAdversarialLoss, DClassificationLoss
from models.utils import GAdversarialLoss, GContentLoss, GStyleLoss, GClassificationLoss


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

        # Blocks for calculating generator losses
        self.g_adv_loss = GAdversarialLoss()
        self.g_class_loss = GClassificationLoss()
        self.g_content_loss = GContentLoss()
        self.g_style_loss = GStyleLoss()

        # Blocks for calculating discriminator losses
        self.d_adv_loss = DAdversarialLoss()
        self.d_class_loss = DClassificationLoss()

        self.train_step = 0

        self.log_image_interval = self.model_configs['log_image_interval']
        self.batch_size = self.dataset_configs['batch_size']

    def training_step(self, batch, batch_idx, optimizer_idx):

        content_image, (style_image, style_classes) = batch

        # Taking logs each self.log_image_interval iterations
        if optimizer_idx == 0 and self.train_step % self.log_image_interval == 0:
            _, _, stylized_image, _ = self.generator(content_image, style_image, style_classes)

            grid_content_images = make_grid(content_image,
                                            nrow=self.batch_size,
                                            normalize=True)
            grid_style_images = make_grid(style_image,
                                          nrow=self.batch_size,
                                          normalize=True)

            grid_stylized_images = make_grid(stylized_image,
                                             nrow=self.batch_size,
                                             normalize=True)

            self.logger.experiment.add_image("Content images: Train stage", grid_content_images, self.train_step)
            self.logger.experiment.add_image("Style images: Train stage", grid_style_images, self.train_step)
            self.logger.experiment.add_image("Stylized images: Train stage", grid_stylized_images, self.train_step)

        # Generator step
        if optimizer_idx == 0:
            self.train_step += 1

            gen_loss = self.generator_loss(content_image, style_image, style_classes)
            self.log('gen_loss', gen_loss)

            return gen_loss

        # Discriminator step
        else:
            d_loss = self.discriminator_loss(content_image, style_image, style_classes)
            self.log('d_loss', d_loss)

            return d_loss

    def generator_loss(self, content_image, style_image, labels):
        """
        Calculates the total loss of discriminator.

        min L_G = L_A + λ_DS * L_{DS} + λ_c L_c + λ_s * L_s

        @param content_image: images with content
        @param style_image: image with style
        @param labels: style classes
        @return: loss
        """
        lambda_adversarial = self.model_configs['lambda_adversarial']
        lambda_style_class = self.model_configs['lambda_style_classification']
        lambda_content = self.model_configs['lambda_content']
        lambda_style = self.model_configs['lambda_style']

        encoded_content_image, encoded_style_image, stylized_image, encoded_stylized_image = \
            self.generator(content_image, style_image)

        discriminator_real_prob, discriminator_real_styles = self.discriminator(stylized_image)

        g_adv = lambda_adversarial * self.g_adv_loss(discriminator_real_prob)
        g_class = lambda_style_class * self.g_class_loss(discriminator_real_prob, discriminator_real_styles, labels)
        g_content = lambda_content * self.g_content_loss(stylized_image, encoded_stylized_image)
        g_style = lambda_style * self.g_style_loss(encoded_stylized_image, encoded_style_image)

        self.log('g_adv', g_adv)
        self.log('g_class', g_class)
        self.log('g_content', g_content)
        self.log('g_style', g_style)

        return g_adv + g_class + g_content + g_style

    def discriminator_loss(self, content_image, style_image, labels):
        lambda_style_class = self.model_configs['lambda_style_classification']

        encoded_content_image, encoded_style_image, stylized_image, encoded_stylized_image = \
            self.generator(content_image, style_image)

        # Passing fake image (generated)
        discriminator_real_prob, discriminator_real_classes = self.discriminator(stylized_image)

        # Passing real image (style)
        discriminator_style_prob, discriminator_style_classes = self.discriminator(style_image)

        d_adv = self.d_adv_loss(discriminator_real_prob, discriminator_style_classes)
        d_class = lambda_style_class * self.d_class_loss(discriminator_real_classes,
                                                         discriminator_style_classes,
                                                         labels)

        self.log('d_adv', d_adv)
        self.log('d_class', d_class)

        return d_adv + d_class

    def configure_optimizers(self):
        lr = self.model_configs['lr_start']
        betas = (self.model_configs['adam_b1'], self.model_configs['adam_b2'])
        step_size = self.model_configs['lr_step_size']

        gen_opt = Adam(self.generator.parameters(), lr=lr, betas=betas)
        discr_opt = Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        # Here should be linear decay, but in torch there is no linear decay
        gen_lr_sched = StepLR(gen_opt, step_size=step_size, gamma=0.5)
        discr_lr_sched = StepLR(discr_opt, step_size=step_size, gamma=0.5)

        return [gen_opt, discr_opt], [gen_lr_sched, discr_lr_sched]

import pytorch_lightning as pl

# This file contains all additional blocks, losses.
# Losses should be also blocks(modules).
import torch
import torch.nn as nn


class Mask(pl.LightningModule):
    """
    Some comments of how it works...
    """

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class AdaIN(pl.LightningModule):
    """
    IdaIN (Adaptive instance normalization)
    """

    def __init__(self):
        super().__init__()

    def mean(self, x):
        """
        Takes a $(n,c,h,w)$ tensor as input and returns the average across it's spatial dimensions as (h,w) tensor
        """
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        """
        Takes a (n,c,h,w) tensor as input and returns the standard deviation across it's spatial dimensions as (h,w) tensor
        """
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mean(x)).permute([2, 3, 0, 1]) ** 2, (2, 3)) + 0.000000023) / (
                    x.shape[2] * x.shape[3]))

    def forward(self, x, y):
        """
        Takes a content embedding x and a style embedding y and changes transforms
        the mean and standard deviation of the content embedding to that of the style
        """
        return (self.sigma(y) * ((x.permute([2, 3, 0, 1]) - self.mean(x)) / self.sigma(x)) + self.mean(y)).permute(
            [2, 3, 0, 1])


class DecoderBlock(pl.LightningModule):

    def __init__(self, in_channels, out_channels, k_size, padding):
        """
        Additional block for constructing Decoder NN

        @param in_channels: number of input channels for nn.Conv2d
        @param out_channels: number of output channels for nn.Conv2d
        @param k_size: kernel size
        @param padding: size of padding
        """
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),

            # TODO(Why here is Relu inplace?)
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),

            # TODO(Why here is Relu inplace?)
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.block(x)


class GramMatrix(pl.LightningModule):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.shape()
        features = x.view(batch_size * channels, height * width)  # resize F_XL into \hat F_XL
        gram = torch.mm(features, features.t())  # compute the gram product

        return gram.div(batch_size * channels * height * width)


class AdversarialLoss(pl.LightningModule):
    """
    $L_A = E[log Prob(Real|D(x̂))]$
    """

    def __init__(self):
        super(AdversarialLoss, self).__init__()
        pass

    def forward(self, discriminator_real_prob: torch.Tensor):
        return torch.log(discriminator_real_prob).mean()


class ClassificationLoss(pl.LightningModule):
    """
    L_{DS} = E [log Prob(s|D(x̂))]
    """

    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def forward(self, discriminator_real_prob, discriminator_real_classes, labels):
        """
        Implementation inspires by
        https://github.com/jianpingliu/AC-GAN/blob/master/ac_gan.py


        @param discriminator_real_prob: probability of real image
        @param discriminator_real_classes: probability of style class
        @param labels: initial classes of images
        @return: DS loss
        """
        return torch.log(discriminator_real_classes[labels]).mean()


class ContentLoss(pl.LightningModule):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, generated_image_content, generated_image_style):
        return (generated_image_content[3] - generated_image_style[3]).abs().mean()


class StyleLoss(pl.LightningModule):
    """
    L_s = E[ ∑ \abs(Gram(y(l)) − Gram(x̂(l))) ].
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.gram = GramMatrix()

    def forward(self, generated_image, generated_style):
        loss = torch.tensor([
                                self.gram(generated_style[i]) - self.gram(generated_image[i])
                            ] for i in range(4))

        return loss.abs().mean()

import pytorch_lightning as pl
import torch.nn as nn


# Realization of Discriminator (all addition modules should be in utils.py)
class Discriminator(pl.LightningModule):
    """
    Implementation of Discriminator, followed by
    https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/.

    6-layer architecture, input is assumed to be (256, 256).
    """

    def __init__(self, relu_negative_slope, verbose=False):
        """

        :param relu_negative_slope: relu slope, read from config
        :param verbose: To see output of x.shape when forwarding input
        """
        super(Discriminator, self).__init__()

        neg_slope = relu_negative_slope
        self.verbose = verbose

        # Traditional 6-layer architecture
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        self.layer5 = nn.Sequential(
            # In article here kernel_size=4 , but dimensions won't match in other case (needed [-1, 512, 16, 16])
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        # Output layer, like linear, should be (1*16*16)
        self.layer6 = nn.Sequential(
            # In article here kernel_size=4 , but dimensions won't match in other case (needed [-1, 512, 16, 16])
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.verbose:
            print(f"x shape={x.shape}")

        # assert x.shape[1:] == (3, 256, 256), "Only pictures with shape [-1, 3, 256, 256] are supported"

        x = self.layer1(x)
        if self.verbose:
            print(f"Size of x={x.shape} should be [-1, 64, 128, 128]")

        x = self.layer2(x)
        if self.verbose:
            print(f"Size of x={x.shape} should be [-1, 128, 64, 64]")

        x = self.layer3(x)
        if self.verbose:
            print(f"Size of x={x.shape} should be [-1, 256, 32, 32]")

        x = self.layer4(x)
        if self.verbose:
            print(f"Size of x={x.shape} should be [-1, 512, 16, 16]")

        x = self.layer5(x)
        if self.verbose:
            print(f"Size of x={x.shape} should be [-1, 512, 16, 16]")

        x = self.layer6(x)
        if self.verbose:
            print(f"Size of x={x.shape} should be [-1, 1, 16, 16]")

        return x

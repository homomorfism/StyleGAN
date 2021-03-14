import pytorch_lightning as pl
import torch
import torch.nn as nn


# Realization of Discriminator (all addition modules should be in utils.py)
class Discriminator(pl.LightningModule):
    """
    Implementation of Discriminator, followed by
    https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/.
    6-layer architecture, input is assumed to be (256, 256).
    """

    def __init__(self, relu_negative_slope, style_classes, verbose=False):
        """
        :param relu_negative_slope: relu slope, read from config
        :param verbose: To see output of x.shape when forwarding input (only once)
        """
        super(Discriminator, self).__init__()

        neg_slope = relu_negative_slope
        self.verbose = verbose
        self.style_classes = style_classes

        # Traditional 6-layer architecture
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        self.layer5 = nn.Sequential(
            # In article here kernel_size=4 , but dimensions won't match in other case (needed [-1, 512, 16, 16])
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=neg_slope),
        )

        self.probabilities = nn.Sequential(
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )

        self.styles = nn.Sequential(
            nn.Linear(in_features=512, out_features=style_classes),
            nn.Softmax()
        )

    def forward(self, x):
        """
        Forward propagation with one-time displaying x shapes transformation (if verbose=True)
        :param x: input tensor
        :return: processes tensor
        """

        if self.verbose:
            print(f"x shape={x.shape}")

        assert x.shape[1:] == torch.Size([3, 256, 256]), "Only pictures with shape [-1, 3, 256, 256] are supported"

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

        # Converting to [-1, 512] tensor
        x = nn.AvgPool2d(kernel_size=16)(x)
        flatten = nn.Flatten()(x)

        assert flatten.shape[1:] == torch.Size([512]), f"The shape of x: {flatten.shape}, but expected: [-1, 512]!"

        probabilities = self.probabilities(flatten)
        if self.verbose:
            print(f"Size of probabilities={probabilities.shape} should be [-1, 1]")

        styles = self.style_classes(flatten)
        if self.verbose:
            print(f"Size of style classes={styles.shape} should be [-1, {self.style_classes}]")

        if self.verbose:
            # Displaying additional info only once
            self.verbose = False

        return probabilities, styles

    def display_gradients(self):
        """
        Displays gradients of each layer just to make sure that all gradients are non-Null, for debugging purposes.
        :return: None
        """
        print("printing gradients of each layer...")

        for i, layer in enumerate(
                [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.probabilities, self.styles]):
            for j, module in enumerate(layer.parameters()):
                print(f"Layer: {i}, module: {j}, grad: {module.grad}")
                assert module.grad is not None, "This gradient in this module is None!"

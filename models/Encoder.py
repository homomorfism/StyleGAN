import pytorch_lightning as pl
import torch.nn as nn
from torchvision import models


class Encoder(pl.LightningModule):

    def __init__(self, verbose=False):
        """

        @param verbose: flag that prints only ones the shapes
        """
        super(Encoder, self).__init__()
        model = models.vgg16(pretrained=True)
        self.verbose = verbose

        features = list(model.features.children())[:31:]
        self.block1 = nn.Sequential(*features[:5:])
        self.pooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(8, 8), dilation=(1, 1), ceil_mode=False)

        self.block2 = nn.Sequential(*features[5:10:])
        self.pooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(4, 4), dilation=(1, 1), ceil_mode=False)

        self.block3 = nn.Sequential(*features[10:17:])
        self.pooling3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)

        self.block4 = nn.Sequential(*features[17:24:])

    def forward(self, x):

        if self.verbose:
            print(f"Encoder, init shape: {x.shape()}")

        x = self.block1(x)
        to_concat1 = self.pooling1(x)

        x = self.block2(x)
        to_concat2 = self.pooling2(x)

        x = self.block3(x)
        to_concat3 = self.pooling3(x)

        x = self.block4(x)

        output = [to_concat1, to_concat2, to_concat3, x]
        if self.verbose:
            self.verbose = False

        return output

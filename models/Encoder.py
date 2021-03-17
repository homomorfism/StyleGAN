import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models


class Encoder(pl.LightningModule):

    def __init__(self):
        super(Encoder, self).__init__()
        model = models.vgg16(pretrained=True)

        features = list(model.features.children())[:31:]
        self.block1 = nn.Sequential(*features[:5:])
        self.pooling1 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(8, 8), dilation=(1, 1), ceil_mode=False, return_indices=True
        )

        self.block2 = nn.Sequential(*features[5:10:])
        self.pooling2 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(4, 4), dilation=(1, 1), ceil_mode=False, return_indices=True
        )

        self.block3 = nn.Sequential(*features[10:17:])
        self.pooling3 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False, return_indices=True
        )

        self.block4 = nn.Sequential(*features[17:24:])

    def forward(self, x):

        x = self.block1(x)
        to_concat1, indexes1 = self.pooling1(x)

        x = self.block2(x)
        to_concat2, indexes2 = self.pooling2(x)

        x = self.block3(x)
        to_concat3, indexes3 = self.pooling3(x)

        x = self.block4(x)

        output = torch.cat((to_concat1, to_concat2, to_concat3, x), dim=2)
        return output, [indexes1, indexes2, indexes3]

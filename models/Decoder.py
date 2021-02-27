import torch.nn as nn
from torchvision import models
import numpy as np
import torch
import pytorch_lightning as pl
from torchsummary import summary


class Decoder(pl.LightningModule):
  def __init__(self, in_channels, middle_channels, out_channels, pooling_indices):
    super(Decoder, self).__init__()
    self.pooling_indices = pooling_indices
    self.block1 = nn.Sequential(
        nn.Conv2d(in_channels, middle_channels[0], kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(middle_channels[0]),
        nn.ReLU(inplace=True),
        nn.Conv2d(middle_channels[0], middle_channels[0], kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(middle_channels[0]),
        nn.ReLU(inplace=True),
        # nn.ConvTranspose2d(middle_channels, middle_channels, kernel_size=(3, 3), stride=(2, 2)),
    )
    self.unpool1 = nn.MaxUnpool2d(2, stride=2)

    self.block2 = nn.Sequential(
        nn.Conv2d(middle_channels[0], middle_channels[1], kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(middle_channels[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(middle_channels[1], middle_channels[1], kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(middle_channels[1]),
        nn.ReLU(inplace=True),
        # nn.ConvTranspose2d(middle_channels, middle_channels, kernel_size=(3, 3), stride=(2, 2)),
    )
    self.unpool2 = nn.MaxUnpool2d(2, stride=2)

    self.block3 = nn.Sequential(
        nn.Conv2d(middle_channels[1], middle_channels[2], kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(middle_channels[2]),
        nn.ReLU(inplace=True),
        nn.Conv2d(middle_channels[2], middle_channels[2], kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(middle_channels[2]),
        nn.ReLU(inplace=True),
        # nn.ConvTranspose2d(middle_channels, middle_channels, kernel_size=(3, 3), stride=(2, 2)),
    )
    self.unpool3 = nn.MaxUnpool2d(2, stride=2)

    self.block4 = nn.Sequential(
        nn.Conv2d(middle_channels[2], middle_channels[3], kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(middle_channels[3]),
        nn.ReLU(inplace=True),
        nn.Conv2d(middle_channels[3], middle_channels[3], kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(middle_channels[3]),
        nn.ReLU(inplace=True),
        # nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=(3, 3), stride=(2, 2)),
    )
    self.unpool4 = nn.MaxUnpool2d(2, stride=2)

    self.block5 = nn.Sequential(
        nn.Conv2d(middle_channels[3], out_channels, kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    x = self.block1(x)
    print('Shape after block1: ', x.shape)
    x = self.unpool1(x, self.pooling_indices[0])

    x = self.block2(x)
    print('Shape after block2: ', x.shape)
    x = self.unpool2(x, self.pooling_indices[1])

    x = self.block3(x)
    print('Shape after block3: ', x.shape)
    x = self.unpool3(x, self.pooling_indices[2])

    x = self.block4(x)
    print('Shape after block4: ', x.shape)
    x = self.unpool4(x, self.pooling_indices[3])

    output = self.block5(x)

    return output
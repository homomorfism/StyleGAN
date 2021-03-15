import pytorch_lightning as pl
import torch.nn as nn

from models.utils import DecoderBlock


class Decoder(pl.LightningModule):

    def __init__(self, in_channels, middle_channels, out_channels, pooling_indices, verbose=False):
        super(Decoder, self).__init__()
        self.pooling_indices = pooling_indices
        self.verbose = verbose

        self.block1 = DecoderBlock(
            in_channels=in_channels,
            out_channels=middle_channels[0],
            k_size=(3, 3),
            padding=(1, 1)
        )

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)

        self.block2 = DecoderBlock(
            in_channels=middle_channels[0],
            out_channels=middle_channels[1],
            k_size=(3, 3),
            padding=(1, 1)
        )

        self.unpool2 = nn.MaxUnpool2d(2, stride=2)

        self.block3 = DecoderBlock(
            in_channels=middle_channels[1],
            out_channels=middle_channels[2],
            k_size=(3, 3),
            padding=(1, 1)
        )

        self.unpool3 = nn.MaxUnpool2d(2, stride=2)

        self.block4 = DecoderBlock(
            in_channels=middle_channels[2],
            out_channels=middle_channels[3],
            k_size=(3, 3),
            padding=(1, 1)
        )

        self.unpool4 = nn.MaxUnpool2d(2, stride=2)

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=middle_channels[3], out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block1(x)

        if self.verbose:
            print('Shape after block1: ', x.shape)

        x = self.unpool1(x, self.pooling_indices[0])

        x = self.block2(x)

        if self.verbose:
            print('Shape after block2: ', x.shape)
        x = self.unpool2(x, self.pooling_indices[1])

        x = self.block3(x)
        if self.verbose:
            print('Shape after block3: ', x.shape)
        x = self.unpool3(x, self.pooling_indices[2])

        x = self.block4(x)
        if self.verbose:
            print('Shape after block4: ', x.shape)
            self.verbose = False

        x = self.unpool4(x, self.pooling_indices[3])

        output = self.block5(x)

        return output

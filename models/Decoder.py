import pytorch_lightning as pl
import torch.nn as nn

from models.utils import DecoderBlock


class Decoder(pl.LightningModule):

    def __init__(self, in_channels, middle_channels, out_channels):
        # TODO(Middle channels find info about them)
        super(Decoder, self).__init__()

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

    def forward(self, x, pooling_indices):

        # In encoder we stack indexes from the biggest image to small -> we need to invert list of indexes
        pooling_indices = pooling_indices[::-1]

        x = self.block1(x)
        x = self.unpool1(x, pooling_indices[0])

        x = self.block2(x)
        x = self.unpool2(x, pooling_indices[1])

        x = self.block3(x)
        x = self.unpool3(x, pooling_indices[2])

        x = self.block4(x)
        x = self.unpool4(x, pooling_indices[3])

        output = self.block5(x)

        return output

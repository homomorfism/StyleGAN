import pytorch_lightning as pl
import torch.nn as nn


# How to run
'''
middle_channels = [256, 128, 64, 32]
in_channels = 960
out_channels = 1
decoder = Decoder(in_channels, middle_channels, out_channels, indexes)
'''

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

        # In encoder we stack indexes from the biggest image to small -> we need to invert list of indexes
        pooling_indices = self.pooling_indices
        # print('x.shape: ', x.shape)

        x = self.block1(x)
        # print('block1.shape: ', x.shape)
        x = self.unpool1(x, pooling_indices[2])
        # print('unpool1.shape: ', x.shape)

        x = self.block2(x)
        # print('block2.shape: ', x.shape)
        x = self.unpool2(x, pooling_indices[1])
        # print('unpool2.shape: ', x.shape)

        x = self.block3(x)
        x = self.unpool3(x, pooling_indices[0])
        # print('block3.shape: ', x.shape)

        x = self.block4(x)
        # print('block4.shape: ', x.shape)

        output = self.block5(x)

        return output

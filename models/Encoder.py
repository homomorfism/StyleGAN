import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models


class Encoder(pl.LightningModule):

    def __init__(self):
        super(Encoder, self).__init__()
        model = models.vgg16(pretrained=True)

        features = list(model.features.children())[:31:]
        self.block1 = nn.Sequential(*features[:5:])
        self.pooling1 = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2), stride=(8, 8), dilation=(1, 1), ceil_mode=False))

        self.block2 = nn.Sequential(*features[5:10:])
        self.pooling2 = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2), stride=(4, 4), dilation=(1, 1), ceil_mode=False))

        self.block3 = nn.Sequential(*features[10:17:])
        self.pooling3 = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False))

        self.block4 = nn.Sequential(*features[17:24:])

    def forward(self, x):
        print('Initial: ', x.shape)

        x = self.block1(x)
        to_concat1 = self.pooling1(x)
        # print('After block1: ', x.shape)
        print('After pooling1: ', to_concat1.shape)

        x = self.block2(x)
        to_concat2 = self.pooling2(x)
        # print('After block2: ', x.shape)
        print('After pooling2: ', to_concat2.shape)

        x = self.block3(x)
        to_concat3 = self.pooling3(x)
        # print('After block3: ', x.shape)
        print('After pooling3: ', to_concat3.shape)

        x = self.block4(x)
        # print('After block4: ', x.shape)

        output = torch.cat((to_concat1, to_concat2, to_concat3, x), 1)
        # print('Output: ', output.shape)
        return output


def encoder_summary(model, input_size):
    print(summary(model, input_size))


def encoder_testing():
    tensor = torch.randn([64, 3, 256, 256])
    encoder = Encoder()
    encoder(tensor.float())


if __name__ == '__main__':
    # encoder = Encoder()
    # encoder_summary(encoder, (3, 256, 256))
    encoder_testing()

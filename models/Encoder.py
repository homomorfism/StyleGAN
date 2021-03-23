import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models


class Encoder(pl.LightningModule):

    def __init__(self):
        super(Encoder, self).__init__()
        model = models.vgg16(pretrained=True)

        features = list(model.features.children())[:31:]
        self.block1 = nn.Sequential(*features[:4:])
        self.next_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        self.cat_pooling1 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(8, 8), dilation=(1, 1), ceil_mode=False
        )

        self.block2 = nn.Sequential(*features[5:9:])
        self.next_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        self.cat_pooling2 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(4, 4), dilation=(1, 1), ceil_mode=False
        )

        self.block3 = nn.Sequential(*features[10:16:])
        self.next_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        self.cat_pooling3 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
        )

        self.block4 = nn.Sequential(*features[17:23:])
        # self.next_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

    def forward(self, x):

        x = self.block1(x)
        to_concat1 = self.cat_pooling1(x)
        print('to_concat1.shape: ', to_concat1.shape)

        x, indexes1 = self.next_pooling1(x)
        print('indexes1.shape: ', x.shape)

        
        x = self.block2(x)
        to_concat2 = self.cat_pooling2(x)
        print('to_concat2.shape: ', to_concat2.shape)

        x, indexes2 = self.next_pooling2(x)
        print('indexes2.shape: ', x.shape)


        x = self.block3(x)
        to_concat3 = self.cat_pooling3(x)
        print('to_concat3.shape: ', to_concat3.shape)

        x, indexes3 = self.next_pooling3(x)
        print('indexes3.shape: ', x.shape)

        x = self.block4(x)
        # x, indexes4 = self.next_pooling4(x)
        # print('indexes4.shape: ', x.shape)


        output = torch.cat((to_concat1, to_concat2, to_concat3, x), dim=1)
        return output, [indexes1, indexes2, indexes3]

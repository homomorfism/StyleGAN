import torch.nn as nn
from torchvision import models
import numpy as np
import torch
import pytorch_lightning as pl
from torchsummary import summary


class AdaIN(pl.LightningModule):
    def __init__(self):
        super().__init__()

    """ Takes a (n,c,h,w) tensor as input and returns the average across it's spatial dimensions as (h,w) tensor"""
    def mu(self, x):
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    """ Takes a (n,c,h,w) tensor as input and returns the standard deviation across it's spatial dimensions as (h,w) tensor"""
    def sigma(self, x):
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    """ Takes a content embeding x and a style embeding y and changes transforms the mean and standard deviation of the content embedding to that of the style"""
    def forward(self, x, y):
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])
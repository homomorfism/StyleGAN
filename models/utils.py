import pytorch_lightning as pl


# This file contains all additional blocks, losses.
# Losses should be also blocks(modules).

class Mask(pl.LightningModule):
    """
    Some comments of how it works...
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class AdaIN(pl.LightningModule):
    """
    Some comments of how it works...
    """
    def __init__(self):
        super(AdaIN, self).__init__()
        pass

    def forward(self, x, y):
        pass

import pytorch_lightning as pl
import torch

from models.Decoder import Decoder
from models.Encoder import Encoder
from models.utils import Mask, AdaIN


class Generator(pl.LightningModule):

    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = Encoder()

        # Output shape from encoder: [-1, 512, 16, 16]
        self.decoder = Decoder(
            in_channels=512,
            middle_channels=[],
            out_channels=3,
            pooling_indices=[],
            verbose=False
        )

        self.mask = Mask()
        self.ada_in = AdaIN()

    def forward(self, content_image, style_image):
        assert content_image.shape[1:] == torch.Size([3, 256, 256]), \
            "only content image sizes [-1, 3, 256, 256] are supported!"

        assert style_image.shape[1:] == torch.Size([3, 256, 256]), \
            "only style image sizes [-1, 3, 256, 256] are supported!"

        encoded_content_image = self.encoder(content_image)
        encoded_style_image = self.encoder(style_image)

        ada_in_normalized = self.ada_in(encoded_style_image, encoded_content_image)
        mask_normalized = self.mask(encoded_style_image, encoded_content_image)

        # z = M(x, y) × x + (1 − M(x, y)) × A(x, y),
        combined_image = mask_normalized * encoded_content_image + (1 - mask_normalized) * ada_in_normalized

        decoded_image = self.decoder(combined_image)
        encoded_combined_image = self.encoder(decoded_image)

        return encoded_content_image, encoded_style_image, decoded_image, encoded_combined_image

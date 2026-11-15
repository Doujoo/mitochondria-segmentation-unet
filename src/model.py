import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    The basic block of the U-Net: (Conv 3x3 -> BatchNorm -> ReLU) two times.
    With padding=1 the output has the same height and width as the input.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net (Ronneberger et al., 2015) for binary segmentation.

    Shape of the tensors for an input (B, 1, 256, 256):

        input  (1, 256)                                           output (1, 256)
          |                                                            ^
        down1  (32, 256) --------------- skip ---------------->  up1 (32, 256)
          | maxpool                                                    ^ upconv
        down2  (64, 128) --------------- skip ---------------->  up2 (64, 128)
          | maxpool                                                    ^ upconv
        down3  (128, 64) --------------- skip ---------------->  up3 (128, 64)
          | maxpool                                                    ^ upconv
        down4  (256, 32) --------------- skip ---------------->  up4 (256, 32)
          | maxpool                                                    ^ upconv
          +------------------------> bottleneck (512, 16) -------------+

    (channels, height): each maxpool divides the size by 2, each upconv multiplies it by 2.
    Height and width of the input must be divisible by 16.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Encoder
        self.down1 = DoubleConv(in_channels, c)
        self.down2 = DoubleConv(c, 2 * c)
        self.down3 = DoubleConv(2 * c, 4 * c)
        self.down4 = DoubleConv(4 * c, 8 * c)
        self.bottleneck = DoubleConv(8 * c, 16 * c)

        # Decoder: each upconv doubles the size and halves the channels,
        # then the skip is concatenated, so the DoubleConv gets twice the channels
        self.upconv4 = nn.ConvTranspose2d(16 * c, 8 * c, kernel_size=2, stride=2)
        self.up4 = DoubleConv(16 * c, 8 * c)
        self.upconv3 = nn.ConvTranspose2d(8 * c, 4 * c, kernel_size=2, stride=2)
        self.up3 = DoubleConv(8 * c, 4 * c)
        self.upconv2 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=2, stride=2)
        self.up2 = DoubleConv(4 * c, 2 * c)
        self.upconv1 = nn.ConvTranspose2d(2 * c, c, kernel_size=2, stride=2)
        self.up1 = DoubleConv(2 * c, c)

        # 1 output channel: probability of mitochondria for each pixel
        self.final_conv = nn.Conv2d(c, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A batch of images. Shape: (B, 1, H, W)

        Returns:
            torch.Tensor: Probability map between 0 and 1. Shape: (B, 1, H, W)
        """
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))

        # dim=1 is the channel dimension
        u4 = self.up4(torch.cat([d4, self.upconv4(b)], dim=1))
        u3 = self.up3(torch.cat([d3, self.upconv3(u4)], dim=1))
        u2 = self.up2(torch.cat([d2, self.upconv2(u3)], dim=1))
        u1 = self.up1(torch.cat([d1, self.upconv1(u2)], dim=1))

        return self.sigmoid(self.final_conv(u1))

import torch

from src.model import DoubleConv, UNet


def test_double_conv_shape():
    block = DoubleConv(1, 32)
    x = torch.randn(2, 1, 64, 64)
    assert block(x).shape == (2, 32, 64, 64)


def test_unet_output_shape():
    """The output must have the same size as the input, with 1 channel."""
    model = UNet()
    x = torch.randn(2, 1, 128, 128)
    output = model(x)
    assert output.shape == (2, 1, 128, 128)


def test_unet_output_is_probability():
    model = UNet()
    output = model(torch.randn(1, 1, 64, 96))
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)

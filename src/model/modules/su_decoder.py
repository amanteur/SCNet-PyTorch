from typing import List, Tuple

import torch
import torch.nn as nn

from model.utils import get_convtranspose_output_padding


class FusionLayer(nn.Module):
    def __init__(
        self, input_dim: int, kernel_size: int = 3, stride: int = 1, padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim * 2,
            input_dim * 2,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
        )
        self.activation = nn.GLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, F, T, C) (B, F, T, C)
        Output: (B, F, T, C)
        """
        x = x1 + x2
        x = x.repeat(1, 1, 1, 2)
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.activation(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, stride: int, output_padding: int
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            input_dim, output_dim, 1, (stride, 1), output_padding=(output_padding, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C_in, F, T)
        Output: (B, C_out, F * stride + output_padding, T)
        """
        return self.conv(x)


class SULayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        upsample_stride: int,
        subband_shape: int,
        sd_interval: Tuple[int, int],
    ):
        super().__init__()
        sd_shape = sd_interval[1] - sd_interval[0]
        upsample_output_padding = get_convtranspose_output_padding(
            input_shape=sd_shape, output_shape=subband_shape, stride=upsample_stride
        )
        self.upsample = Upsample(
            input_dim=input_dim,
            output_dim=output_dim,
            stride=upsample_stride,
            output_padding=upsample_output_padding,
        )
        self.sd_interval = sd_interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, F, T, C)
        Output: (B, F, T, C)
        """
        x = x[:, self.sd_interval[0]: self.sd_interval[1]]
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)
        return x


class SUBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        upsample_strides: List[int],
        subband_shapes: List[int],
        sd_intervals: List[Tuple[int, int]],
    ):
        super().__init__()
        self.fusion_layer = FusionLayer(input_dim=input_dim)
        self.su_layers = nn.ModuleList(
            SULayer(
                input_dim=input_dim,
                output_dim=output_dim,
                upsample_stride=uss,
                subband_shape=sbs,
                sd_interval=sdi,
            )
            for i, (uss, sbs, sdi) in enumerate(
                zip(upsample_strides, subband_shapes, sd_intervals)
            )
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, Fi, T, Ci)
        Input: (B, Fi-1, T, Ci-1)
        """
        x = self.fusion_layer(x, x_skip)
        x = torch.concat([layer(x) for layer in self.su_layers], dim=1)
        return x

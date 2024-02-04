from typing import List, Tuple

import torch
import torch.nn as nn

from model.utils import create_intervals


class Downsample(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 1, (stride, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C_in, F, T)
        Output: (B, C_out, F // stride, T)
        """
        return self.conv(x)


class ConvolutionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_sizes: List[int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=input_dim),
            nn.Conv1d(
                input_dim,
                2 * hidden_dim,
                kernel_sizes[0],
                stride=1,
                padding=(kernel_sizes[0] - 1) // 2,
                bias=bias,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_sizes[1],
                stride=1,
                padding=(kernel_sizes[1] - 1) // 2,
                groups=hidden_dim,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(
                hidden_dim,
                input_dim,
                kernel_sizes[2],
                stride=1,
                padding=(kernel_sizes[2] - 1) // 2,
                bias=bias,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, D)
        Output: (B, T, D)
        """
        x = x.transpose(1, 2)
        x = x + self.sequential(x)
        x = x.transpose(1, 2)
        return x


class SDLayer(nn.Module):
    def __init__(
        self,
        subband_interval: Tuple[float, float],
        input_dim: int,
        output_dim: int,
        downsample_stride: int,
        n_conv_modules: int,
        kernel_sizes: List[int],
        bias: bool = True,
    ):
        super().__init__()
        self.subband_interval = subband_interval
        self.downsample = Downsample(input_dim, output_dim, downsample_stride)
        self.activation = nn.GELU()
        conv_modules = [
            ConvolutionModule(
                input_dim=output_dim,
                hidden_dim=output_dim // 4,
                kernel_sizes=kernel_sizes,
                bias=bias,
            )
            for _ in range(n_conv_modules)
        ]
        self.conv_modules = nn.Sequential(*conv_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, Fi, T, Ci)
        Output: (B, Fi+1, T, Ci+1)
        """
        B, F, T, C = x.shape
        x = x[:, int(self.subband_interval[0] * F): int(self.subband_interval[1] * F)]
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1)

        B, F, T, C = x.shape
        x = x.reshape((B * F), T, C)
        x = self.conv_modules(x)
        x = x.reshape(B, F, T, C)

        return x


class SDBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bandsplit_ratios: List[float],
        downsample_strides: List[int],
        n_conv_modules: List[int],
        kernel_sizes: List[int],
    ):
        super().__init__()
        assert sum(bandsplit_ratios) == 1, "The split ratios must sum up to 1."
        subband_intervals = create_intervals(bandsplit_ratios)
        self.sd_layers = nn.ModuleList(
            SDLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                subband_interval=sbi,
                downsample_stride=dss,
                n_conv_modules=ncm,
                kernel_sizes=kernel_sizes,
            )
            for sbi, dss, ncm in zip(
                subband_intervals, downsample_strides, n_conv_modules
            )
        )
        self.global_conv2d = nn.Conv2d(output_dim, output_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: (B, Fi, T, Ci)
        Input: (B, Fi+1, T, Ci+1)
        """
        x_skip = torch.concat([layer(x) for layer in self.sd_layers], dim=1)
        x = self.global_conv2d(x_skip.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x, x_skip

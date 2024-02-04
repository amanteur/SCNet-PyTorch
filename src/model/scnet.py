from typing import List

import torch
import torch.nn as nn

from model.modules import DualPathRNN, SDBlock, SUBlock
from model.utils import compute_sd_layer_shapes, compute_gcr


class SCNet(nn.Module):
    def __init__(
        self,
        freq_dim: int,
        input_dims: List[int],
        output_dims: List[int],
        bandsplit_ratios: List[float],
        downsample_strides: List[int],
        n_conv_modules: List[int],
        n_rnn_layers: int,
        rnn_hidden_dim: int,
        n_sources: int = 4,
    ):
        super().__init__()
        self.assert_input_data(
            input_dims,
            output_dims,
            bandsplit_ratios,
            downsample_strides,
            n_conv_modules,
        )

        n_blocks = len(input_dims)
        subband_shapes, sd_intervals = compute_sd_layer_shapes(
            input_shape=freq_dim,
            bandsplit_ratios=bandsplit_ratios,
            downsample_strides=downsample_strides,
            n_layers=len(input_dims),
        )

        self.sd_blocks = nn.ModuleList(
            SDBlock(
                input_dim=input_dims[i],
                output_dim=output_dims[i],
                bandsplit_ratios=bandsplit_ratios,
                downsample_strides=downsample_strides,
                n_conv_modules=n_conv_modules,
            )
            for i in range(0, n_blocks)
        )
        self.dualpath_block = DualPathRNN(
            n_layers=n_rnn_layers,
            input_dim=output_dims[-1],
            hidden_dim=rnn_hidden_dim,
        )
        self.su_blocks = nn.ModuleList(
            SUBlock(
                input_dim=output_dims[i],
                output_dim=input_dims[i] if i != 0 else input_dims[i] * n_sources,
                subband_shapes=subband_shapes[i],
                sd_intervals=sd_intervals[i],
                upsample_strides=downsample_strides,
            )
            for i in range(n_blocks - 1, -1, -1)
        )
        self.gcr = compute_gcr(subband_shapes)

    @staticmethod
    def assert_input_data(*args):
        for arg1 in args:
            for arg2 in args:
                if len(arg1) != len(arg2):
                    raise ValueError(
                        f"Shapes of input features {arg1} and {arg2} are not equal."
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, F, T, C), where ...
        Output: (B, F, T, C, S), where ...
        """
        B, F, T, C = x.shape

        # encoder part
        x_skips = []
        for sd_block in self.sd_blocks:
            x, x_skip = sd_block(x)
            x_skips.append(x_skip)

        # separation part
        x = self.dualpath_block(x)

        # decoder part
        for su_block, x_skip in zip(self.su_blocks, reversed(x_skips)):
            x = su_block(x, x_skip)

        # split into N sources
        x = x.reshape(B, F, T, C, -1)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    net_params = {
        "freq_dim": 2049,
        "input_dims": [4, 32, 64],
        "output_dims": [32, 64, 128],
        "bandsplit_ratios": [.175, .392, .433],
        "downsample_strides": [1, 4, 16],
        "n_conv_modules": [3, 2, 1],
        "n_rnn_layers": 6,
        "rnn_hidden_dim": 128,
        "n_sources": 4,
    }
    device = "cpu"
    B, F, T, C = 4, 2049, 474, 4

    net = SCNet(**net_params).to(device)
    _ = net.eval()

    test_input = torch.rand(B, F, T, C).to(device)
    out = net(test_input)


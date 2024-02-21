from typing import List

import torch
import torch.nn as nn

from model.modules import DualPathRNN, SDBlock, SUBlock
from model.utils import compute_sd_layer_shapes, compute_gcr


class SCNet(nn.Module):
    """
    SCNet class implements a source separation network,
    which explicitly split the spectrogram of the mixture into several subbands
    and introduce a sparsity-based encoder to model different frequency bands.

    Paper: "SCNET: SPARSE COMPRESSION NETWORK FOR MUSIC SOURCE SEPARATION"
    Authors: Weinan Tong, Jiaxu Zhu et al.
    Link: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - n_fft (int): Number of FFTs to determine the frequency dimension of the input.
    - dimes (List[int]): List of channel dimensions for each block.
    - bandsplit_ratios (List[float]): List of ratios for splitting the frequency bands.
    - downsample_strides (List[int]): List of stride values for downsampling in each block.
    - n_conv_modules (List[int]): List specifying the number of convolutional modules in each block.
    - n_rnn_layers (int): Number of recurrent layers in the dual path RNN.
    - rnn_hidden_dim (int): Dimensionality of the hidden state in the dual path RNN.
    - n_sources (int, optional): Number of sources to be separated. Default is 4.

    Shapes:
    - Input: (B, F, T, C) where
        B is batch size,
        F is the number of features,
        T is sequence length,
        C is input dimensionality.
    - Output: (B, F, T, C, S) where
        B is batch size,
        F is the number of features,
        T is sequence length,
        C is input dimensionality,
        S is the number of sources.
    """

    def __init__(
        self,
        n_fft: int,
        dims: List[int],
        bandsplit_ratios: List[float],
        downsample_strides: List[int],
        n_conv_modules: List[int],
        n_rnn_layers: int,
        rnn_hidden_dim: int,
        n_sources: int = 4,
    ):
        """
        Initializes SCNet with input parameters.
        """
        super().__init__()
        self.assert_input_data(
            bandsplit_ratios,
            downsample_strides,
            n_conv_modules,
        )

        n_blocks = len(dims) - 1
        n_freq_bins = n_fft // 2 + 1
        subband_shapes, sd_intervals = compute_sd_layer_shapes(
            input_shape=n_freq_bins,
            bandsplit_ratios=bandsplit_ratios,
            downsample_strides=downsample_strides,
            n_layers=n_blocks,
        )
        self.sd_blocks = nn.ModuleList(
            SDBlock(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                bandsplit_ratios=bandsplit_ratios,
                downsample_strides=downsample_strides,
                n_conv_modules=n_conv_modules,
            )
            for i in range(n_blocks)
        )
        self.dualpath_blocks = DualPathRNN(
            n_layers=n_rnn_layers,
            input_dim=dims[-1],
            hidden_dim=rnn_hidden_dim,
        )
        self.su_blocks = nn.ModuleList(
            SUBlock(
                input_dim=dims[i + 1],
                output_dim=dims[i] if i != 0 else dims[i] * n_sources,
                subband_shapes=subband_shapes[i],
                sd_intervals=sd_intervals[i],
                upsample_strides=downsample_strides,
            )
            for i in reversed(range(n_blocks))
        )
        self.gcr = compute_gcr(subband_shapes)

    @staticmethod
    def assert_input_data(*args):
        """
        Asserts that the shapes of input features are equal.
        """
        for arg1 in args:
            for arg2 in args:
                if len(arg1) != len(arg2):
                    raise ValueError(
                        f"Shapes of input features {arg1} and {arg2} are not equal."
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the SCNet.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, F, T, C).

        Returns:
        - torch.Tensor: Output tensor of shape (B, F, T, C, S).
        """
        B, F, T, C = x.shape

        # encoder part
        x_skips = []
        for sd_block in self.sd_blocks:
            x, x_skip = sd_block(x)
            x_skips.append(x_skip)

        # separation part
        x = self.dualpath_blocks(x)

        # decoder part
        for su_block, x_skip in zip(self.su_blocks, reversed(x_skips)):
            x = su_block(x, x_skip)

        # split into N sources
        x = x.reshape(B, F, T, C, -1)

        return x

    def count_parameters(self):
        """
        Counts the total number of parameters in the SCNet.
        """
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    net_params = {
        "n_fft": 4096,
        "dims": [4, 32, 64, 128],
        "bandsplit_ratios": [0.175, 0.392, 0.433],
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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate


class Separator(nn.Module):
    """
    Neural Network Separator.

    This class implements a neural network-based separator for audio source separation.

    Args:
    - cfg (DictConfig): Configuration object containing separator settings.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize Separator.
        """
        super().__init__()
        self.cfg = cfg
        self.return_spec = cfg.return_spec

        self.stft = instantiate(cfg.transform.stft)
        self.net = instantiate(cfg.net)
        self.istft = instantiate(cfg.transform.istft)

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Pad input tensor to fit STFT requirements.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - x (torch.Tensor): Padded input tensor.
        - pad_size (int): Size of padding.
        """
        hop_length = self.cfg.transform.stft.hop_length
        pad_size = hop_length - x.shape[-1] % hop_length
        x = F.pad(x, (0, pad_size))
        return x, pad_size

    def apply_stft(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Apply Short-Time Fourier Transform (STFT) to input tensor.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - x (torch.Tensor): Transformed tensor.
        - pad_size (int): Size of padding applied.
        """
        x, pad_size = self.pad(x)
        x = self.stft(x)
        x = torch.view_as_real(x)
        return x, pad_size

    def apply_istft(
        self, x: torch.Tensor, pad_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply Inverse Short-Time Fourier Transform (ISTFT) to input tensor.

        Args:
        - x (torch.Tensor): Input tensor.
        - pad_size (int): Size of padding applied.

        Returns:
        - x (torch.Tensor): Inverse transformed tensor.
        """
        x = torch.view_as_complex(x)
        x = self.istft(x)
        if pad_size is not None:
            x = x[..., :-pad_size]
        return x

    def apply_net(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply neural network to input tensor.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - x (torch.Tensor): Transformed tensor.

        Shapes:
        - Input: (B, Ch, Fr, T, Co) where
            B is batch size,
            Ch is the number of channels,
            Fr is the number of frequencies,
            T is sequence length,
            Co is real/imag part.
        - Output: (B, S, Ch, Fr, T, Co) where
            B is batch size,
            S is number of separated sources,
            Ch is the number of channels,
            Fr is the number of frequencies,
            T is sequence length,
            Co is real/imag part.
        """
        B, Ch, Fr, T, Co = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(B, Fr, T, Ch * Co)

        x = self.net(x)

        S = x.shape[-1]
        x = x.reshape(B, F, T, Ch, Co, S).permute(0, 5, 3, 1, 2, 4).contiguous()
        return x

    def forward(self, x_mixture: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the separator.

        Args:
        - x_mixture (torch.Tensor): Input mixture tensor.

        Returns:
        - torch.Tensor: Separated sources.
        """
        spec_mixture, pad_size = self.apply_stft(x_mixture)

        spec_sources = self.apply_net(spec_mixture)

        if self.return_spec:
            return spec_sources

        x_sources = self.apply_istft(spec_sources, pad_size)
        return x_sources

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig


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

        self.bs = cfg.batch_size
        self.sr = cfg.sample_rate
        self.ws = int(cfg.window_size * cfg.sample_rate)
        self.ss = int(cfg.step_size * cfg.sample_rate)
        self.ps = self.ws - self.ss

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
        x = x.reshape(B, Fr, T, Ch, Co, S).permute(0, 5, 3, 1, 2, 4).contiguous()
        return x

    def forward(self, wav_mixture: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the separator.

        Args:
        - wav_mixture (torch.Tensor): Input mixture tensor.

        Returns:
        - torch.Tensor: Separated sources.

        Shapes:
        input (B, Ch, T)
        -> stft (B, Ch, F, T, Co)
        -> net (B, S, Ch, F, T, Co)
        -> istft (B, S, Ch, T)
        """
        spec_mixture, pad_size = self.apply_stft(wav_mixture)

        spec_sources = self.apply_net(spec_mixture)

        wav_sources = self.apply_istft(spec_sources, pad_size)

        if self.return_spec:
            return wav_sources, spec_sources
        return wav_sources, None

    def pad_whole(self, y: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        """
        # pad for overlap-add
        padding_add = self.ss - (y.shape[-1] + self.ps * 2 - self.ws) % self.ss
        y = F.pad(
            y,
            (self.ps, self.ps + padding_add),
            'constant'
        )
        return y, padding_add

    def unpad_whole(self, y: torch.Tensor, padding_add: int) -> torch.Tensor:
        """
        """
        return y[..., self.ps:-(self.ps + padding_add)]

    def unfold(self, y: torch.Tensor) -> torch.Tensor:
        """
        """
        y = y.unfold(
            -1,
            self.ws,
            self.ss
        ).permute(1, 0, 2)
        return y

    def fold(self, y_chunks: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        overlap-add
        """
        n_chunks, n_sources, *_ = y_chunks.shape
        y_out = torch.zeros_like(y).unsqueeze(0).repeat(n_sources, 1, 1)
        start = 0
        for i in range(n_chunks):
            y_out[..., start:start + self.ws] += y_chunks[i]
            start += self.ss
        return y_out

    def forward_batches(self, y_chunks: torch.Tensor) -> torch.Tensor:
        """
        """
        norm_value = self.ws / self.ss
        y_chunks = torch.cat(
            [
                self(y_chunks[start:start + self.bs])[0] / norm_value
                for start in range(0, y_chunks.shape[0], self.bs)
            ]
        )
        return y_chunks

    @torch.no_grad()
    def separate(self, y: torch.Tensor) -> torch.Tensor:
        """
        """
        y, padding_add = self.pad_whole(y)
        y_chunks = self.unfold(y)

        y_chunks = self.forward_batches(y_chunks)

        y = self.fold(y_chunks, y)
        y = self.unpad_whole(y, padding_add)

        return y

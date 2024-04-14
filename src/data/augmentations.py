# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Data augmentations taken and adapted from https://github.com/adefossez/demucs/blob/main/demucs/augment.py.
"""

import random
import torch
from torch import nn


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """

    def __init__(self, shift: int = 8192, same: bool = False):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                srcs = 1 if self.same else sources
                offsets = torch.randint(
                    self.shift, [batch, srcs, 1, 1], device=wav.device
                )
                offsets = offsets.expand(-1, sources, channels, -1)
                indexes = torch.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    """

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = torch.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1, -1, -1, time)
            right = 1 - left
            wav = torch.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    """

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = torch.randint(
                2, (batch, sources, 1, 1), device=wav.device, dtype=torch.float32
            )
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """

    def __init__(self, proba: float = 1.0, group_size: int = 4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separately. This allows to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
        """
        super().__init__()
        self.proba = proba
        self.group_size = group_size

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        batch, streams, channels, time = wav.size()
        device = wav.device

        if self.training and random.random() < self.proba:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(
                    f"Batch size {batch} must be divisible by group size {group_size}"
                )
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = torch.argsort(
                torch.rand(groups, group_size, streams, 1, 1, device=device), dim=1
            )
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class Scale(nn.Module):
    def __init__(
        self, proba: bool = 1.0, min_scale: bool = 0.25, max_scale: bool = 1.25
    ):
        super().__init__()
        self.proba = proba
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = torch.empty(batch, streams, 1, 1, device=device).uniform_(
                self.min_scale, self.max_scale
            )
            wav *= scales
        return wav

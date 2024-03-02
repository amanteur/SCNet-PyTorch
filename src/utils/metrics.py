from typing import Any

import torch
from torchmetrics.audio import SignalDistortionRatio


def global_signal_distortion_ratio(
    preds: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Calculates the Global Signal Distortion Ratio (GSDR) between predicted and target signals.

    Args:
    - preds (torch.Tensor): Predicted signal tensor.
    - target (torch.Tensor): Target signal tensor.
    - epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-7.

    Returns:
    - torch.Tensor: Mean Global Signal Distortion Ratio (GSDR) over the batch.
    """
    num = torch.sum(
        torch.square(target), dim=(-2, -1)
    ) + epsilon
    den = torch.sum(
        torch.square(target - preds), dim=(-2, -1)
    ) + epsilon
    usdr = 10 * torch.log10(num / den)
    return usdr.mean()


class GlobalSignalDistortionRatio(SignalDistortionRatio):
    """
    Computes the Global Signal Distortion Ratio (GSDR) metric for audio signals
    as it was described firstly in Sound Demixing Challenge.

    This metric calculates the ratio between the energy of the original signal
    and the energy of the difference between the original and the predicted signal,
    measured in decibels (dB).

    Paper: https://arxiv.org/pdf/2308.06979.pdf
    """
    def __init__(
        self,
        epsilon: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the GlobalSignalDistortionRatio metric.
        """
        super().__init__(**kwargs)

        self.epsilon = epsilon

        self.add_state("sum_sdr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        sdr_batch = global_signal_distortion_ratio(
            preds, target, self.epsilon
        )

        self.sum_sdr += sdr_batch.sum()
        self.total += sdr_batch.numel()

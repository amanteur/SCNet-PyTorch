import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error Loss for Complex Inputs.

    Args:
        - eps (float): A small value to prevent division by zero.
    """

    def __init__(self, eps: float = 1e-6):
        """
        Initialize RMSELoss.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of RMSELoss.

        Args:
        - pred (torch.Tensor): Predicted values, a tensor of shape (batch_size, ..., 2),
            where last dimension is real/imaginary part.
        - target (torch.Tensor): Target values, a tensor of shape (batch_size, ..., 2),
            where last dimension is real/imaginary part.

        Returns:
        - loss (torch.Tensor): Computed RMSE loss.
        """
        loss = torch.sqrt(
            self.mse(pred[..., 0], target[..., 0])
            + self.mse(pred[..., 1], target[..., 1])
            + self.eps
        )
        return loss

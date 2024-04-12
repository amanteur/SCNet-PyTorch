from typing import Dict

import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig

from model.separator import Separator


class LightningWrapper(LightningModule):
    """
    This class serves as a LightningModule for training and evaluating the source separation model.

    Args:
    - cfg (DictConfig): Configuration object containing wrapper settings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the LightningWrapper.
        """
        super().__init__()

        self.cfg = cfg

        self.sep = Separator(**cfg.separator)

        self.loss = instantiate(cfg.loss)
        self.optimizer = instantiate(cfg.optimizer, params=self.sep.parameters())
        self.scheduler = instantiate(cfg.scheduler) if cfg.get("scheduler") else None
        self.metrics = nn.ModuleDict(instantiate(cfg.metrics))

        self.save_hyperparameters(cfg)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, mode="train")
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, mode="val")
        return loss

    def step(self, batch: torch.Tensor, mode: str = "train") -> torch.Tensor:
        """
        Common step function for training and validation.

        Args:
        - batch (torch.Tensor): Batch of input data.
        - mode (str): Mode of operation. Defaults to 'train'.

        Returns:
        - torch.Tensor: Loss tensor.
        """
        wav_mix, wav_src = batch["mixture"], batch["sources"]
        wav_src_hat, spec_src_hat = self.sep(wav_mix)
        spec_src, _ = self.sep.apply_stft(wav_src)

        loss = self.loss(spec_src_hat, spec_src)

        self.log(f"{mode}/loss", loss.detach(), prog_bar=True)

        if mode == "val":
            metrics = self.compute_metrics(wav_src_hat, wav_src)
            self.log_dict(metrics)
        return loss

    @torch.no_grad()
    def compute_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes metrics for evaluation.

        Args:
        - preds (torch.Tensor): Predicted sources tensor.
        - target (torch.Tensor): Target sources tensor.

        Returns:
        - Dict[str, torch.Tensor]: Dictionary containing computed metrics.
        """
        metrics = {}
        for key in self.metrics:
            for i, source in enumerate(self.cfg.dataset.sources):
                metrics[f"val/{key}_{source}"] = self.metrics[key](
                    preds[:, i], target[:, i]
                )
            metrics[f"val/{key}"] = self.metrics[key](preds, target)
        return metrics

    def on_before_optimizer_step(self, *args, **kwargs) -> None:
        norms = grad_norm(self, norm_type=2)
        norms = dict(filter(lambda elem: "_total" in elem[0], norms.items()))
        self.log_dict(norms)
        return

    def configure_optimizers(self):
        return [self.optimizer]

    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics()
        items.pop("v_num", None)
        return items

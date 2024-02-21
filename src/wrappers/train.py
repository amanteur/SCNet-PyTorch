from typing import Optional, Tuple

import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from model.separator import Separator


class TrainProgram:
    def __init__(self, cfg: DictConfig):
        """ """
        self.mode = "train"
        self.cfg = cfg

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.model.device == "cuda"
            else "cpu"
        )

        self.model = Separator(cfg.model).to(self.device)

        (self.train_dataloader, self.val_dataloader) = self.initialize_dataloaders(
            cfg.data, mode=self.mode
        )

        (
            self.loss,
            self.optimizer,
            self.scheduler,
        ) = self.initialize_training_components(cfg.model)

    @staticmethod
    def initialize_dataloaders(
        cfg: DictConfig, mode: str = "train"
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """ """
        dataset = instantiate(cfg.dataset)
        if mode == "train":
            # TODO: add logic to process test datasets
            pass
        if cfg.use_validation:
            train_dataset, val_dataset = dataset.get_train_val_split(**cfg.split)
            train_dataloader = DataLoader(train_dataset, **cfg.loader.train)
            val_dataloader = DataLoader(val_dataset, **cfg.loader.validation)
        else:
            train_dataloader = DataLoader(dataset, **cfg.loader.train)
            val_dataloader = None
        return train_dataloader, val_dataloader

    def initialize_training_components(
        self,
        cfg: DictConfig,
    ):
        """ """
        loss = instantiate(cfg.loss)
        optimizer = instantiate(cfg.optimizer, params=self.model.parameters())
        scheduler = instantiate(cfg.scheduler) if cfg.scheduler is not None else None
        return loss, optimizer, scheduler

import logging
import traceback
from shutil import rmtree

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader

from utils.lightning import LightningWrapper

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>...")
        seed_everything(cfg.seed, workers=True)

    log.info("Initializing DataLoaders...")
    dataset = instantiate(cfg.dataset)
    if cfg.use_validation:
        train_dataset, val_dataset = dataset.get_train_val_split(**cfg.train_val_split)
        train_dataloader = DataLoader(train_dataset, **cfg.loader.train)
        val_dataloader = DataLoader(val_dataset, **cfg.loader.validation)
    else:
        train_dataloader = DataLoader(dataset, **cfg.loader.train)
        val_dataloader = None

    log.info("Initializing LightningWrapper...")
    lt_wrapper = LightningWrapper(cfg)

    log.info("Initializing training utilities...")
    logger = instantiate(cfg.logger)
    callbacks = list(instantiate(cfg.callbacks).values())

    log.info("Initializing trainer...")
    trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    log.info("Starting training...")
    try:
        trainer.fit(
            lt_wrapper,
            train_dataloader,
            val_dataloader
        )
    except Exception as e:
        log.error(f"Finished with error:\n{traceback.format_exc()}")

    if cfg.trainer.fast_dev_run:
        rmtree(cfg.output_dir)

    log.info("Training finished!")


if __name__ == "__main__":
    train()

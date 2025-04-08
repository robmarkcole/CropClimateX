from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
import os
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor

home_folder = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
data_dir = home_folder / "data/CropClimateX/"
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you: asd
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    visualizations,
)

log = RankedLogger(__name__, rank_zero_only=True)


def run(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _convert_="all")

    # log.info("Instantiating loggers...")
    # logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    logger = []
    # log.info("Instantiating callbacks...")
    # callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"), len(logger))
    callbacks = []

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, _convert_="object")

    # log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)
    if cfg.data.split.nr_folds != len(cfg.ckpt_paths):
        raise ValueError("Number of folds must match number of checkpoint paths!")

    ys, preds, meta = [], [], []
    for i, ckpt_path in enumerate(cfg.ckpt_paths):
        # setup dataloader with different split (to get correct ids)
        cfg.data.split.k = i
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, _convert_="all")
        datamodule.setup()
        # extract the y and geoids from the dataset
        predictions = trainer.predict(model=model, dataloaders=datamodule.val_dataloader(), ckpt_path=ckpt_path)

        all_predictions = []
        all_metadata = []
        all_labels = []
        for batch_preds in predictions:
            all_predictions.append(batch_preds["preds"])
            all_metadata.extend(batch_preds["meta"])
            all_labels.append(batch_preds["y"])
        # Combine into single tensors
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # add it to global
        ys.append(all_labels)
        preds.append(all_predictions)
        meta.extend(all_metadata)

    # combine all predictions and ys (meta is already combined)
    y = torch.cat(ys, dim=0)
    y_hat = torch.cat(preds, dim=0)
    # adjust the y and pred to the corn range:
    min, max = 1.2427706792199058, 17.392512889486664
    y = y * (max - min) + min
    y_hat = y_hat * (max - min) + min
    fig = visualizations.plot_county_residuals(y, y_hat, meta, os.path.join(data_dir,"county_list.geojson"), plot_lines='states')
    os.makedirs("results", exist_ok=True)
    fig.savefig(f'results/county_residuals_cv{cfg.data.split.nr_folds}.png', bbox_inches='tight', dpi=400)


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print(cfg)
    run(cfg)

if __name__ == "__main__":
    main()

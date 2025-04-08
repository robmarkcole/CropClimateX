from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import optuna
from joblib import parallel_backend
from functools import reduce

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
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
)

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def train(cfg: DictConfig, trial) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, _convert_="all")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _convert_="all")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    callbacks += [optuna.integration.PyTorchLightningPruningCallback(trial, monitor=cfg.monitor_metric)]

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, _convert_="all")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

def update_cfg(trial: optuna.trial.Trial, cfg_section, search_space_section, cfg_total, search_space_total, round_0=True):
    """Recursively suggest values and update the config."""
    for key, value in search_space_section.items():
        if isinstance(value, (dict, DictConfig)) and "type" not in value:
            # Recurse into nested sections
            update_cfg(trial, cfg_section[key], value, cfg_total, search_space_total, round_0)
        else:
            if value["type"] == "same":
                if round_0:
                    continue
                else:
                    # use the same value as in the other parameter
                    value = reduce(lambda d, k: d[k], value["name"].split('.'), cfg_total)
                    suggestion = value
            elif value["type"] == "dependend":
                if round_0:
                    continue
                else:
                    # get the value from the other parameter
                    set_value = reduce(lambda d, k: d[k], value["name"].split('.'), cfg_total)
                    possible_values = reduce(lambda d, k: d[k], value["name"].split('.'), search_space_total)
                    if not 'choices' in possible_values:
                        raise ValueError("Only categorical values are supported for dependend parameters")
                    # get the idx of the value
                    idx = possible_values['choices'].index(set_value)
                    # get the suggested value
                    options = value["choices"]
                    if isinstance(options, list):
                        suggestion = options[idx]
                    else:
                        func = getattr(trial, f"suggest_{options[idx]['type']}")
                        value_list = dict(filter(lambda item: item[0] != 'type', options[idx].items()))
                        suggestion = func(key, **value_list)
            else:
                if not round_0:
                    continue
                # use optuna to suggest a value
                func = getattr(trial, f"suggest_{value['type']}")
                value_list = dict(filter(lambda item: item[0] != 'type', value.items()))
                suggestion = func(key, **value_list)

            # Update the corresponding section of the config
            cfg_section[key] = suggestion


def tune(cfg: DictConfig):
    # init the study
    log.info(f"Instantiating study <{cfg.study._target_}>")
    study = hydra.utils.instantiate(cfg.study, _convert_="all")

    search_space = hydra.utils.instantiate(cfg.search_space, _convert_="all")

    # define the objective function
    def objective(trial: optuna.trial.Trial) -> float:
        """objective to optimize by optuna."""
        cfg_trial = cfg.copy()
        update_cfg(trial, cfg_trial, search_space, None, None, round_0=True)
        update_cfg(trial, cfg_trial, search_space, cfg_trial, search_space, round_0=False)

        metrics, _ = train(cfg_trial, trial=trial)
        return metrics[cfg.monitor_metric]

    log.info(f"Starting optimization")
    optimize = hydra.utils.instantiate(cfg.optimize, _convert_="all")
    study.optimize(objective, **optimize)

    log.info("Number of finished trials: {}".format(len(study.trials)))

    log.info("Best trial:")
    trial = study.best_trial

    log.info("  Value: {}".format(trial.value))

    log.info("  Params: ")
    for key, value in trial.params.items():
        log.info("    {}: {}".format(key, value))
    return trial

@hydra.main(version_base="1.3", config_path="../configs", config_name="tune.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # tune the model
    _ = tune(cfg)


if __name__ == "__main__":
    main()

from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig
from lightning.pytorch.callbacks import LearningRateMonitor

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig, logger_len=0) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="all"))

    if logger_len == 0:
        size = len(callbacks)
        callbacks = [cb for cb in callbacks if not isinstance(cb, LearningRateMonitor)]
        if len(callbacks) < size:
            log.warning("LearningRateMonitor callback requires a logger to be used, LearningRateMonitor was removed.")

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger

def instantiate_pipeline(pipeline: DictConfig, metrics: DictConfig) -> Pipeline:
    """Creates a sklearn pipeline with all the preprocessing steps specified in `pipeline`, ordered in a sequential manner

    :param pipeline (DictConfig): the config containing the instructions for creating the feature selectors or transformers
    :param metrics (DictConfig): the config containing the instructions for creating the metrics
    :return: [sklearn.pipeline.Pipeline]: a pipeline with all the preprocessing steps, in a sequential manner
    """
    steps = []

    for k,v in pipeline.items():
        # instantiate the pipeline step, and append to the list of steps
        pipeline_step = (k, hydra.utils.instantiate(v))
        steps.append(pipeline_step)

    # create metrics as scorers
    metrics = {k: hydra.utils.instantiate(v) for k,v in metrics.items()}

    return Pipeline(steps), metrics
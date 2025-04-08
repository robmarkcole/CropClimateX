from typing import Any, Dict, Tuple

import torch
import torchmetrics
from lightning import LightningModule
import abc
import einops
import wandb
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class LightningWrapper(LightningModule):
    """Wrapper to use for torch module to integrate with lightning.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        output_activation_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        loss_agg: dict[str, torchmetrics.Metric],
        metrics: Dict[str, torchmetrics.Metric],
        best_metric_agg: torchmetrics.Metric,
        compile: bool,
        scheduler: torch.optim.lr_scheduler=None,
        sample_split: dict=None,
        data_dims: str='',
        **kwargs: Any
    ) -> None:
        """Initialize.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        if kwargs:
            log.warning(f"Unused kwargs in lightning module: {kwargs}")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.loss = loss

        # metric objects for calculating and averaging accuracy across batches
        # transform the omega config to normal dict and metrics
        self.train_metrics = torchmetrics.MetricCollection(metrics['train'])
        self.val_metrics = torchmetrics.MetricCollection(metrics['val'])
        self.test_metrics = torchmetrics.MetricCollection(metrics['test'])

        # for averaging loss across batches
        self.train_loss = loss_agg['train']
        self.val_loss = loss_agg['val']
        self.test_loss = loss_agg['test']

        self.best_val_metric = best_metric_agg['metric']
        self.output_activation_function = output_activation_function
        self.data_dims = data_dims
        self.sample_split = sample_split

    def forward(self, *args) -> torch.Tensor:
        return self.net(*args)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metrics.reset()
        self.best_val_metric.reset()

    @abc.abstractmethod
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def log_metrics(self, loss, dic, prefix, batch_size):
        # update and log metrics
        # log tensor metrics seperatly
        metric_dict_1, metric_dict_2 = {}, {}
        for k, v in dic.items():
            if v.ndimension() > 0:
                if v.ndimension() > 1:
                    raise ValueError(f"Metric {k} has more than one dimension. It will not be logged.")
                for i, v_ in enumerate(v):
                    metric_dict_2[f"{prefix}/{k}_{i}"] = v_
            else:
                metric_dict_1[f"{prefix}/{k}"] = v
        self.log_dict(metric_dict_1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(metric_dict_2, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, *preds = self.model_step(batch)
        dic = self.train_metrics(*preds)
        self.log_metrics(loss, dic, "train", batch_size=len(batch['x']))

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, *preds = self.model_step(batch)
        dic = self.val_metrics(*preds)
        self.log_metrics(loss, dic, "val", batch_size=len(batch['x']))

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        loss, *preds = self.model_step(batch)
        return {'preds': preds[0], 'y': preds[1], 'meta': batch['meta']}

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        super().on_validation_epoch_end() # needed to call original method
        score = self.val_metrics[self.hparams.best_metric_agg['metric_name']].compute() # select the choosen metric and compute
        self.best_val_metric(score)  # update best so far
        # log as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/best", self.best_val_metric.compute(), sync_dist=True, prog_bar=False)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, *preds = self.model_step(batch)
        dic = self.test_metrics(*preds)
        self.log_metrics(loss, dic, "test", batch_size=len(batch['x']))

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

class YieldLightningWrapper(LightningWrapper):
    def model_step(
        self, batch: dict[str, tuple]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch['x'], batch['y']
        # get size to reverse concat
        p_sizes = [x[i].shape[0] for i in range(len(x))]
        # concatenate in the batch/patch dimension
        x = torch.cat(x, dim=0)
        y = [ys.view(1) for ys in y] # make sure it is a tensor
        y = torch.cat(y, dim=0)
        preds = self.forward(x)
        # disentangle patches
        preds = torch.split(preds, p_sizes)
        # combine the preds and compute loss
        preds = [torch.mean(pred).view(1) for pred in preds]
        preds = torch.cat(preds, dim=0)
        # use final activation function after aggregation
        preds = self.output_activation_function(preds)

        loss = self.loss(preds, y)
        return loss, preds, y

class ExtremesLightningWrapper(LightningWrapper):
    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch['x'], batch['y']
        logits = self.forward(x)
        logits = self.output_activation_function(logits)
        # switch time and pred dimension
        if logits.ndimension() > 2:
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            else:
                # bring class dimension to the second position for loss (assumed to be last)
                logits = einops.rearrange(logits, 'b ... c -> b c ...')

        loss = self.loss(logits, y)
        if logits.ndimension() > 1: # in case it is multi dimensional ce - do mean explicitly
            loss = loss.mean() # do this explicitly, as the ce in multi dim otherwise does not work, see https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/10
        return loss, logits, y

if __name__ == "__main__":
    _ = LightningWrapper(None, None, None, None)

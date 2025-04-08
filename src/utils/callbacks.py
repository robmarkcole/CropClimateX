import torch
import lightning.pytorch as pl
import wandb
import matplotlib.pyplot as plt
import os
import torchmetrics
import pandas as pd
import numpy as np

def todevice(batch, device):
    if isinstance(batch['x'], (list,tuple)):
        for k in batch:
            if not 'meta' in k:
                for i in range(len(batch[k])):
                    batch[k][i] = batch[k][i].to(device)
    else:
        batch = {k: v.to(device) if not 'meta' in k else v for k,v in batch.items()}
    return batch

class ImgLogger(pl.Callback):
    def __init__(self, log_dir:str, plot_func, name="media", num_samples=3, band_dims=None, plot_y_p_x_m=[True,True,True,True], every_n_epochs=1):
        super().__init__()
        self.log_dir = log_dir
        self.num_samples = num_samples
        self.subset = {}
        self.plot_func = plot_func
        # prepare dict
        if band_dims:
            if not isinstance(band_dims, dict):
                band_dims = dict(band_dims)
            for k,v in band_dims.items():
                if not isinstance(v, (list, tuple)):
                    band_dims[k] = list(v)
        self.band_dims = band_dims
        self.name = name
        self.plot_y_p_x_m = plot_y_p_x_m
        self.every_n_epochs = every_n_epochs

    def log_func(self, trainer, model, subset, prefix):
        nr_samples = 0
        for batch in subset:
            # bring to same device as model
            batch = todevice(batch, model.device)
            # predict on the batch input
            preds, y = model.model_step(batch)[1:3]
            preds = preds.to('cpu')
            y = y.to('cpu')
            batch = todevice(batch, 'cpu')
            # unpack the batch, plot and log
            for i, preds_y in enumerate(zip(preds, y)):
                p = preds_y[0]
                _y = preds_y[1]
                # select arguments to plot
                args = [_y if self.plot_y_p_x_m[0] else None,
                        p if self.plot_y_p_x_m[1] else None,
                        batch['x'][i] if 'x' in batch and self.plot_y_p_x_m[2] else None,
                        batch['mask'][i] if 'mask' in batch and self.plot_y_p_x_m[-1] else None]
                
                if self.band_dims: # select the dimensions to plot
                    for k,v in self.band_dims.items():
                        args = [a.index_select(k,torch.tensor(v)) if a is not None and a.ndimension() > k else a for a in args[:-1]] + [args[-1]]

                if 'meta' in batch:
                    if isinstance(batch['meta'], list):
                        meta = batch['meta'][i]
                    else:
                        meta = {k: v[i] for k,v in batch['meta'].items()}
                    name = f"{self.name}_{meta['geoid']}"
                    if 'pid' in meta:
                        name += f"_{meta['pid']}"
                    if 'split_part' in meta:
                        name += f"_{meta['split_part']}"
                    subtitles = None
                    if 'time' in meta:
                        if isinstance(meta['time'], str):
                            name += f"_{meta['time']}"
                        else:
                            subtitles = pd.to_datetime(meta['time'].numpy()).strftime('%Y-%m-%d')
                else:
                    name = f"{self.name}_{i}"
                    subtitles = None

                args = [a.detach().numpy() if a is not None else None for a in args]
                fig = self.plot_func(*args, subtitles=subtitles)

                # log fig
                epoch = trainer.current_epoch        
                if trainer.logger:
                    trainer.logger.log_image(key=f"media/{prefix}/{name}", images=[wandb.Image(fig)], caption=[name])

                # save in log dir
                os.makedirs(os.path.join(self.log_dir, 'media/'), exist_ok=True)
                plt.savefig(os.path.join(self.log_dir, f"media/{prefix}_epoch_{epoch}_{name}.pdf"))
                plt.close()

                nr_samples += 1
                if nr_samples >= self.num_samples:
                    return

    def on_train_epoch_end(self, trainer, model):
        if trainer.current_epoch % self.every_n_epochs == 0:
            dl = trainer.datamodule.train_dataloader()
            subset = dl if not isinstance(dl, (list, tuple)) else dl[0]
            self.log_func(trainer, model, subset, 'train')

    def on_validation_epoch_end(self, trainer, model):
        if trainer.current_epoch % self.every_n_epochs == 0:
            dl = trainer.datamodule.val_dataloader()
            subset = dl if not isinstance(dl, (list, tuple)) else dl[0]
            self.log_func(trainer, model, subset, 'val')

    def on_test_epoch_end(self, trainer, model):
        dl = trainer.datamodule.test_dataloader()
        subset = dl if not isinstance(dl, (list, tuple)) else dl[0]
        self.log_func(trainer, model, subset, 'test')

        
class PredLogger(pl.Callback):
    def __init__(self, log_dir:str, plot_func, name="pred", every_n_epochs=1, provide_meta_data=False, **kwargs):
        super().__init__()
        self.log_dir = log_dir
        self.plot_func = plot_func
        self.name = name
        self.kwargs = kwargs
        self.every_n_epochs = every_n_epochs
        self.provide_meta_data = provide_meta_data

    def log_func(self, trainer, model, dl, prefix):
        preds, y, meta = [], [], []
        for batch in dl:
            batch = todevice(batch, model.device)
            out = model.model_step(batch)
            # predict on the batch input
            preds.append(out[1].to('cpu'))
            y.append(out[2].to('cpu'))
            if self.provide_meta_data:
                meta.append(batch['meta'])

        preds = torch.cat(preds, dim=0)
        y = torch.cat(y, dim=0)

        if self.provide_meta_data:
            meta = np.concatenate(meta, axis=0)
            fig = self.plot_func(y, preds, meta=meta, **self.kwargs)
        else:
            fig = self.plot_func(y, preds, **self.kwargs)

        # log fig
        epoch = trainer.current_epoch
        if trainer.logger:
            trainer.logger.log_image(key=f"media/{prefix}/{self.name}", images=[wandb.Image(fig)], caption=[f"{self.name}"])

        # save in log dir
        os.makedirs(os.path.join(self.log_dir, 'media/'), exist_ok=True)
        plt.savefig(os.path.join(self.log_dir, f"media/{prefix}_epoch_{epoch}_{self.name}.pdf"), bbox_inches='tight')
        plt.close()

    def on_validation_epoch_end(self, trainer, model):
        if trainer.current_epoch % self.every_n_epochs == 0:
            dl = trainer.datamodule.val_dataloader()
            subset = dl if not isinstance(dl, (list, tuple)) else dl[0]
            self.log_func(trainer, model, subset, 'val')

    def on_test_epoch_end(self, trainer, model):
        dl = trainer.datamodule.test_dataloader()
        subset = dl if not isinstance(dl, (list, tuple)) else dl[0]
        self.log_func(trainer, model, subset, 'test')

class WatchLogger(pl.Callback):
    """Callback to watch gradients and parameters with wandb"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs 

    def on_train_start(self, trainer, model):
        wandb.watch(model, **self.kwargs)

class StatsLogger(pl.Callback):
    def __init__(self, dims):
        self.dims = dims # dimensions to calculate statistics over
        self.shape = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch == 0:
            with torch.no_grad():
                x = batch['x']
                if self.shape is None:
                    dims = [i for i in range(x.ndimension()) if i not in self.dims]
                    self.shape = [x.shape[i] for i in dims]
                    self._reset(self.shape, device=x.device)

                batch_mean = torch.mean(x, dim=self.dims)
                batch_var = torch.var(x, dim=self.dims)
                batch_count = torch.tensor(x.shape[self.dims[0]], dtype=torch.float)

                n_ab = self.count + batch_count
                m_a = self.mean * self.count
                m_b = batch_mean * batch_count
                M2_a = self.var * self.count
                M2_b = batch_var * batch_count

                delta = batch_mean - self.mean

                self.mean = (m_a + m_b) / (n_ab)
                # we don't subtract -1 from the denominator to match the standard Numpy/PyTorch variances
                self.var = (M2_a + M2_b + delta ** 2 * self.count * batch_count / (n_ab)) / (n_ab)
                self.count += batch_count
                self.std = torch.sqrt(self.var + 1e-8)

                # update min/max channel wise
                batch_min = torch.amin(x, dim=self.dims)
                batch_max = torch.amax(x, dim=self.dims)
                self.min = torch.minimum(self.min, torch.amin(x, dim=self.dims))
                self.max = torch.maximum(self.max, torch.amax(x, dim=self.dims))

    def on_train_epoch_end(self, trainer, pl_module):
        # log table
        dic = {"min": self.min.to('cpu'), "max" : self.max.to('cpu'), "mean": self.mean.to('cpu'), "var": self.var.to('cpu'), "std": self.std.to('cpu')}
        df = pd.DataFrame(dic)
        table = wandb.Table(dataframe=df)
        if pl_module.logger:
            pl_module.logger.experiment.log({"train/stats": table})
        self.shape = None # reset tensors

    def _reset(self, shape, device):
        self.min = torch.ones(shape, device=device)*torch.inf
        self.max = torch.ones(shape, device=device)*-torch.inf
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = torch.zeros(1, device=device)

# class SaveConfigCallback(pl.Callback):
#     def __init__(self, log_dir):
#         self.log_dir = log_dir

#     def on_train_start(self, trainer, pl_module):
#         hydra_config_path = f"{self.log_dir}.hydra/config.yaml"
#         # wandb.save(hydra_config_path)
#         print(hydra_config_path)
#         trainer.logger.experiment.save(hydra_config_path)

# import weightwatcher as ww
# class WeightWatcherLogger(pl.Callback):
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs

#     def on_train_end(self, trainer, model):
#         watcher = ww.WeightWatcher(model=model)
#         results = watcher.analyze(**self.kwargs)
        
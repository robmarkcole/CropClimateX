from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from sklearn.model_selection import KFold
from src.datasets.utils import get_length_from_ratio
import src.datasets.download as d
import yaml
import os
import pandas as pd
import hashlib
from collections import Counter
from src.datasets.utils import make_weights_for_balanced_classes
from src.datasets.utils import DistributedSamplerWrapper
from src.datasets.yield_dataset import YieldDataset
from src.datasets.minicube_dataset import MinicubeDataset
from src.datasets.tabular_yield_dataset import TabularYieldDataset
from src.datasets.prep_dataset import PrepDataset
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = False,
        seed: int = 42,
        split: dict = {'method':'random', 'ratio': [0.8, 0.1, 0.1], 'nr_folds': 1, 'k': 1},
        transformations: dict = None,
        task: str = 'crop_yield',
        bands: dict = {},
        stratified_sampler_args: dict = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # enable to use all available cores
        if num_workers < 0:
            num_workers = os.cpu_count() + num_workers + 1
            if num_workers < 0:
                num_workers = 0
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.kwargs = kwargs

        # data transformations
        if transformations:
            if 'x' in transformations or 'y' in transformations:
                # if there is x and y in conf -> different transformations
                self.x_transforms = transforms.Compose(transformations.get('x', {}).values())
                self.y_transforms = transforms.Compose(transformations.get('y', {}).values())
            else:
                # same transformations
                self.x_transforms = transforms.Compose(transformations.values())
                self.y_transforms = transforms.Compose(transformations.values())
        else:
            self.x_transforms = None
            self.y_transforms = None

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download the dataset."""
        pass
        # d.download_data_from_huggingface(self.hparams.bands, self.hparams.years, ['05113'], output_folder=self.hparams.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.ds_train and not self.ds_val and not self.ds_test:
            self.dataset = None
            # get the right dataset class
            if 'prep' in self.hparams.task:
                dataset_class = PrepDataset
            elif 'tabular' in self.hparams.task and 'yield' in self.hparams.task:
                dataset_class = TabularYieldDataset
            elif 'yield' in self.hparams.task:
                dataset_class = YieldDataset
            elif 'extremes' in self.hparams.task or 'esf' in self.hparams.task:
                dataset_class = MinicubeDataset
            else:
                raise NotImplementedError(f"Task {self.hparams.task} is not implemented.")

            if 'fixed' in self.hparams.split.get('method', 'fixed') and any(len(ids) > 0 for ids in self.hparams.split.get('ratio', [])):
                if self.hparams.split.get('method') == 'fixed_geoids':
                    # geoid split predefined
                    dss = []
                    for geoids in self.hparams.split.get('ratio', []):
                        if len(geoids) > 0:
                            dss.append(dataset_class(self.hparams.data_dir, x_transform=self.x_transforms, y_transform=self.y_transforms, bands=self.hparams.bands, geoids=geoids, seed=self.hparams.seed, **self.kwargs))
                        else:
                            dss.append(None)
                elif self.hparams.split.get('method') == 'fixed_years':
                    # years split predefined
                    dss = []
                    for years in self.hparams.split.get('ratio', []):
                        if len(years) > 0:
                            dss.append(dataset_class(self.hparams.data_dir, x_transform=self.x_transforms, y_transform=self.y_transforms, bands=self.hparams.bands, years=years, seed=self.hparams.seed, **self.kwargs))
                        else:
                            dss.append(None)
                else:
                    raise NotImplementedError(f"Split method {self.hparams.split.get('method')} is not implemented.")
                self.ds_train, self.ds_val, self.ds_test = dss
            else:
                if not 'geoids' in self.kwargs:
                    raise ValueError("No geoids provided for dataset, please provide the geoids in the config.")
                geoids = self.kwargs.pop('geoids')
                dataset = dataset_class(self.hparams.data_dir, x_transform=self.x_transforms, y_transform=self.y_transforms, bands=self.hparams.bands, seed=self.hparams.seed, geoids=geoids, **self.kwargs)
                if self.hparams.split.get('nr_folds', 1) <= 1:
                    # random splitting
                    if self.hparams.split.get('method', 'random') == 'random':
                        train_ids, val_ids, test_ids = random_split(
                            dataset=geoids,
                            lengths=get_length_from_ratio(len(geoids), self.hparams.split.get('ratio')),
                            generator=torch.Generator().manual_seed(self.hparams.seed),
                        )
                        self.ds_train = dataset_class(self.hparams.data_dir, x_transform=self.x_transforms, y_transform=self.y_transforms, bands=self.hparams.bands, seed=self.hparams.seed, geoids=train_ids, **self.kwargs)
                        self.ds_val = dataset_class(self.hparams.data_dir, x_transform=self.x_transforms, y_transform=self.y_transforms, bands=self.hparams.bands, seed=self.hparams.seed, geoids=val_ids, **self.kwargs)
                        self.ds_test = dataset_class(self.hparams.data_dir, x_transform=self.x_transforms, y_transform=self.y_transforms, bands=self.hparams.bands, seed=self.hparams.seed, geoids=test_ids, **self.kwargs)
                else:
                    # cross-validation
                    if not 0 <= self.hparams.split.get("k") <= self.hparams.split.get("nr_folds")-1:
                        raise ValueError(f"incorrect fold number, provide nr between 0 and {self.hparams.split.get('nr_folds')-1}")
                    if self.hparams.split.get('method', 'random') == 'random':
                        # choose fold to train on
                        kf = KFold(n_splits=self.hparams.split.get("nr_folds"), shuffle=True, random_state=self.hparams.seed)
                        all_splits = [k for k in kf.split(geoids)]
                        train_idx, val_idx = all_splits[self.hparams.split.get("k")]
                        train_ids, val_ids = [geoids[i] for i in train_idx], [geoids[i] for i in val_idx]
                        log.info(f'{self.hparams.split.get("nr_folds")} fold cross-validation, using fold {self.hparams.split.get("k")}')
                        self.ds_train = dataset_class(self.hparams.data_dir, x_transform=self.x_transforms, y_transform=self.y_transforms, bands=self.hparams.bands, geoids=train_ids, seed=self.hparams.seed, **self.kwargs)
                        self.ds_val = dataset_class(self.hparams.data_dir, x_transform=self.x_transforms, y_transform=self.y_transforms, bands=self.hparams.bands, geoids=val_ids, seed=self.hparams.seed, **self.kwargs)
                self.dataset = dataset

        # create stratified sampler when defined in config
        self.train_sampler = None
        if self.hparams.stratified_sampler_args is not None:
            # check if labels are the same (make sure ordering is the same before creating hash)
            if hasattr(self.ds_train, 'labels'):
                df_sorted = self.ds_train.labels.sort_values(by=list(self.ds_train.labels.columns)).reset_index(drop=True)
                data_str = df_sorted.to_csv(index=False)
            elif hasattr(self.ds_train, 'fns'):
                data_str = ''.join(f"{file}:{count}" for file, count in sorted(Counter(self.ds_train.fns).items()))
            hash_value = hashlib.sha256(data_str.encode()).hexdigest()
            name = f"_{self.hparams.stratified_sampler_args['name']}_" if 'name' in self.hparams.stratified_sampler_args else ''
            file_name = os.path.join(self.hparams.data_dir, f'sampler{name}{hash_value}.pkl')
            if os.path.exists(file_name):
                weights = torch.load(file_name)
            else:
                # create weights for samples
                weights = make_weights_for_balanced_classes(self.ds_train, n_classes=self.hparams.stratified_sampler_args['n_classes'],
                                                            batch_size=self.batch_size_per_device, target_name=self.hparams.stratified_sampler_args['dl_target_name'],
                                                            num_workers=self.hparams.num_workers)
                torch.save(weights, file_name)
            log.info(f'Using stratified sampler with weights from: {file_name}')
            if type(self.trainer.strategy).__name__ != "SingleDeviceStrategy":
                log.info('wrapping sampler in DistributedSamplerWrapper')
                # set with wrapper for distributed sampler -> since inherited form DistributedSampler, gets not replaced automaticially by lightning
                self.train_sampler = DistributedSamplerWrapper(torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=self.hparams.stratified_sampler_args['replacement']))
            else:
                self.train_sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=self.hparams.stratified_sampler_args['replacement'])

    def full_dataloader(self) -> DataLoader[Any]:
        """Create and return the full dataloader, meaning all years, all crops, all geoids.

        :return: The full dataloader.
        """
        # init dataset with all geoids
        _kwargs = self.kwargs.copy()
        if 'geoids' in self.kwargs:
            geoids = _kwargs.pop('geoids')
        else:
            geoids = self.ds_train.geoids + self.ds_val.geoids + self.ds_test.geoids
        if self.hparams.task == 'crop_yield':
            ds = YieldDataset(self.hparams.data_dir, self.x_transforms, self.y_transforms, self.hparams.bands, geoids=geoids, **_kwargs)
        elif self.hparams.task == 'extremes' or self.hparams.task == 'esf':
            ds = MinicubeDataset(self.hparams.data_dir, self.x_transforms, self.y_transforms, self.hparams.bands, geoids=geoids, **_kwargs)
        else:
            raise NotImplementedError(f"Task {self.hparams.task} is not implemented.")
        # make sure full dataset has exactly the same labels as the split dataset
        if self.dataset is not None:
            ds = self.dataset
        else:
            ds.labels = pd.concat([self.ds_train.labels, self.ds_val.labels, self.ds_test.labels])
        return DataLoader(
            dataset=ds,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_yield_dl if 'yield' in self.hparams.task else None,
            # multiprocessing_context=get_context('loky')
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
            shuffle=True if not self.train_sampler else False, # deactivate since it would use RandomSampler (is control by our sampler)
            sampler=self.train_sampler,
            collate_fn=collate_yield_dl if 'yield' in self.hparams.task else None,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.ds_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
            shuffle=False,
            collate_fn=collate_yield_dl if 'yield' in self.hparams.task else None,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.ds_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
            shuffle=False,
            collate_fn=collate_yield_dl if 'yield' in self.hparams.task else None,
        )

def collate_yield_dl(batch):
    """collate yield dataset differently, because has different number of minicube in each sample."""
    collated_batch = {}
    
    # get keys from first item
    keys = batch[0].keys()
    
    for key in keys:
        collated_batch[key] = [item[key] for item in batch]
    
    return collated_batch

if __name__ == "__main__":
    _ = DataModule()

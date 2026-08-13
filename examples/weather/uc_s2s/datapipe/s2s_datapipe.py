"""
S2S datapipe: wraps the existing GetDataset in a physicsnemo Datapipe.

The underlying GetDataset logic (HDF5 multi-file reading, land/ocean masking,
boundary data loading) is unchanged. This wrapper only adds the physicsnemo
Datapipe interface so training scripts can use DistributedManager cleanly.
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

from .data_loader_multifiles import GetDataset


class S2SDatapipe(Datapipe):
    """PhysicsNeMo-compatible datapipe for S2S forecasting datasets.

    Parameters
    ----------
    params : YParams or DictConfig
        Configuration object with data_dir, year_start/end, batch_size, etc.
    split : str
        One of 'train' or 'val'.
    dist_manager : DistributedManager, optional
        If provided and distributed, wraps dataset with DistributedSampler.
    num_inferences : int
        Number of inference steps (used for validation dataset).
    validate : bool
        Whether to use validation-mode dataset logic.
    shuffle : bool, optional
        Override shuffle behaviour; defaults to True for train, False for val.
    """

    def __init__(
        self,
        params,
        split: str = "train",
        dist_manager=None,
        num_inferences: int = 0,
        validate: bool = False,
        shuffle: bool = None,
    ):
        super().__init__(meta=DatapipeMetaData(name="S2SDatapipe"))

        train = split == "train"

        if split == "train":
            year_start = params.train_year_start
            year_end = params.train_year_end
        else:
            year_start = params.val_year_start
            year_end = params.val_year_end

        self.dataset = GetDataset(
            params,
            params.data_dir,
            year_start,
            year_end,
            train=train,
            num_inferences=num_inferences,
            validate=validate,
        )

        distributed = dist_manager is not None and dist_manager.distributed
        do_shuffle = train if shuffle is None else shuffle

        if distributed:
            self.sampler = DistributedSampler(self.dataset, shuffle=do_shuffle)
        elif do_shuffle:
            self.sampler = torch.utils.data.RandomSampler(self.dataset)
        else:
            self.sampler = None

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=int(params.batch_size),
            num_workers=params.num_data_workers,
            shuffle=False,
            sampler=self.sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def set_epoch(self, epoch: int):
        """Call before each epoch when using DistributedSampler."""
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    @property
    def constant_boundary_data(self):
        return self.dataset.constant_boundary_data

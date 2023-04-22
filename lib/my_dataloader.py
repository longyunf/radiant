import numpy as np
import random
from collections.abc import Mapping, Sequence
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Sampler
from mmcv.parallel.data_container import DataContainer
from mmcv.runner import get_dist_info


def init_data_loader_from_file(args, dataset_class, ann_file):
    
    args_dataset = {'ann_file': ann_file,
                    'data_root': args.dir_data,
                    'mode': 'train'}
    args_data_loader = {'samples_per_gpu': args.samples_per_gpu,
                        'workers_per_gpu': args.workers_per_gpu,
                        'num_gpus': args.num_gpus,
                        'shuffle': False,
                        'seed': args.seed}
    dataset = dataset_class(**args_dataset)    
    
    data_loader = build_dataloader(dataset, **args_data_loader)
    
    return data_loader


def init_data_loader(args, dataset_class, mode):
    
    if mode == 'train':
        ann_file = args.train_ann_file
        nus_version = 'v1.0-trainval'
        modality = None
        samples_per_gpu = args.samples_per_gpu
        shuffle = True
    elif mode == 'val':
        ann_file = args.val_ann_file
        nus_version = 'v1.0-trainval'
        modality = None
        samples_per_gpu = args.samples_per_gpu
        shuffle = False
    elif mode == 'test':
        assert args.eval_set in ['val', 'test']
        if args.eval_set == 'val':
            ann_file = args.val_ann_file
            nus_version = 'v1.0-trainval'
        elif args.eval_set == 'test':
            ann_file = args.test_ann_file 
            nus_version = 'v1.0-test'
        samples_per_gpu = args.test_samples_per_gpu
        shuffle = False
        if args.eval_mono:
            modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False)
        else:
            modality = None
            
    args_dataset = {'ann_file': ann_file,
                    'data_root': args.dir_data,
                    'modality': modality,
                    'mode': mode,
                    'version': nus_version}
    args_data_loader = {'samples_per_gpu': samples_per_gpu,
                        'workers_per_gpu': args.workers_per_gpu,
                        'num_gpus': args.num_gpus,
                        'shuffle': shuffle,
                        'seed': args.seed}
    dataset = dataset_class(**args_dataset)    
    
    data_loader = build_dataloader(dataset, **args_data_loader)
    
    return data_loader


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     shuffle=True,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training (run across multiple machines).
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()   

    sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
    batch_size = num_gpus * samples_per_gpu  
    num_workers = num_gpus * workers_per_gpu 

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,    
        worker_init_fn=init_fn,  
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)  
        self.group_sizes = np.bincount(self.flag)  
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


def collate(batch, samples_per_gpu=1): 
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.
    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.
    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


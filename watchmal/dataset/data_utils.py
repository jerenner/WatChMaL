"""
Utils for handling creation of dataloaders
"""

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch.utils.data import DataLoader

# generic imports
import numpy as np
import random

# WatChMaL imports
from watchmal.dataset.samplers import DistributedSamplerWrapper

# pyg imports
from torch_geometric.loader import DataLoader as PyGDataLoader


def get_data_loader(dataset, is_graph, batch_size, sampler, num_workers, is_distributed, seed, split_path=None, split_key=None, transforms=None):
    """
    Creates a dataloader given the dataset and sampler configs. The dataset and sampler are instantiated using their
    corresponding configs. If using DistributedDataParallel, the sampler is wrapped using DistributedSamplerWrapper.
    A dataloader is returned after being instantiated using this dataset and sampler.

    Args:
        dataset         ... hydra config specifying dataset object
        is_graph        ... a boolean indicating whether the dataset is graph or not
        batch_size      ... batch size
        sampler         ... hydra config specifying sampler object
        num_workers     ... number of workers to use in dataloading
        is_distributed  ... whether running in multiprocessing mode, used to wrap sampler using DistributedSamplerWrapper
        seed            ... seed used to coordinate samplers in distributed mode
        split_path      ... path to indices specifying splitting of dataset among train/val/test
        split_key       ... string key to select indices
        transforms      ... list of transforms to apply

    Returns: dataloader created with instantiated dataset and (possibly wrapped) sampler
    """
    dataset = instantiate(dataset, transforms=transforms,
                          is_distributed=is_distributed)

    if split_path is not None and split_key is not None:
        split_indices = np.load(split_path, allow_pickle=True)[split_key]
        sampler = instantiate(sampler, split_indices)
    else:
        sampler = instantiate(sampler)

    if is_distributed:
        ngpus = torch.distributed.get_world_size()

        batch_size = int(batch_size/ngpus)

        sampler = DistributedSamplerWrapper(sampler=sampler, seed=seed)

    if is_graph:
        return PyGDataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    else:
        # TODO: added drop_last, should decide if we want to keep this
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, drop_last=False, persistent_workers=True, pin_memory=True)


def get_transformations(transformations, transform_names):
    """
    Returns a list of transformation functions from an object and a list of names of the desired transformations, where
    the object has functions with the given names.

    Parameters
    ----------
    transformations : object containing the transformation functions
    transform_names : list of strings

    Returns
    -------

    """
    if transform_names is not None:
        for transform_name in transform_names:
            assert hasattr(
                transformations, transform_name), f"Error: There is no defined transform named {transform_name}"
        transform_funcs = [getattr(transformations, transform_name)
                           for transform_name in transform_names]
        return transform_funcs
    else:
        return None


def apply_random_transformations(transforms, data, segmented_labels=None):
    """
    Randomly chooses a set of transformations to apply, from a given list of transformations, then applies those that
    were randomly chosen to the data and returns the transformed data.

    Parameters
    ----------
    transforms : list of callable
        List of transformation functions to apply to the data.
    data : array_like
        Data to transform
    segmented_labels
        Truth data in the same format as data, to also apply the same transformation.

    Returns
    -------
    data
        The transformed data.
    """
    if transforms is not None:
        for transformation in transforms:
            if random.getrandbits(1):
                data = transformation(data)
                if segmented_labels is not None:
                    segmented_labels = transformation(segmented_labels)
    return data

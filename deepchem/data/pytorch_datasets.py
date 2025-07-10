import numpy as np
import torch
import torch.distributed as dist

from deepchem.data.datasets import NumpyDataset, DiskDataset, ImageDataset
from typing import Optional


class _TorchNumpyDataset(torch.utils.data.IterableDataset):  # type: ignore

    def __init__(self,
                 numpy_dataset: NumpyDataset,
                 epochs: int,
                 deterministic: bool,
                 batch_size: Optional[int] = None):
        """
        Parameters
        ----------
        numpy_dataset: NumpyDataset
            The original NumpyDataset which you want to convert to PyTorch Dataset
        epochs: int
            the number of times to iterate over the Dataset
        deterministic: bool
            if True, the data is produced in order.  If False, a different random
            permutation of the data is used for each epoch.
        batch_size: int
            the number of samples to return in each batch.  If None, each returned
            value is a single sample.
        """
        self.numpy_dataset = numpy_dataset
        self.epochs = epochs
        self.deterministic = deterministic
        self.batch_size = batch_size

    def __iter__(self):
        n_samples = self.numpy_dataset._X.shape[0]
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            first_sample = 0
            last_sample = n_samples
        else:
            first_sample = worker_info.id * n_samples // worker_info.num_workers
            last_sample = (worker_info.id +
                           1) * n_samples // worker_info.num_workers
        for epoch in range(self.epochs):
            if self.deterministic:
                order = first_sample + np.arange(last_sample - first_sample)
            else:
                # Ensure that every worker will pick the same random order for each epoch.
                random = np.random.RandomState(epoch)
                order = random.permutation(n_samples)[first_sample:last_sample]
            if self.batch_size is None:
                for i in order:
                    yield (self.numpy_dataset._X[i], self.numpy_dataset._y[i],
                           self.numpy_dataset._w[i], self.numpy_dataset._ids[i])
            else:
                for i in range(0, len(order), self.batch_size):
                    indices = order[i:i + self.batch_size]
                    yield (self.numpy_dataset._X[indices],
                           self.numpy_dataset._y[indices],
                           self.numpy_dataset._w[indices],
                           self.numpy_dataset._ids[indices])


class _TorchDiskDataset(torch.utils.data.IterableDataset):  # type: ignore

    def __init__(self,
                 disk_dataset: DiskDataset,
                 epochs: int,
                 deterministic: bool,
                 batch_size: Optional[int] = None):
        """
        Parameters
        ----------
        disk_dataset: DiskDataset
            The original DiskDataset which you want to convert to PyTorch Dataset
        epochs: int
            the number of times to iterate over the Dataset
        deterministic: bool
            if True, the data is produced in order.  If False, a different random
            permutation of the data is used for each epoch.
        batch_size: int
            the number of samples to return in each batch.  If None, each returned
            value is a single sample.
        """
        self.disk_dataset = disk_dataset
        self.epochs = epochs
        self.deterministic = deterministic
        self.batch_size = batch_size

    def __len__(self):
        return len(self.disk_dataset)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Determine which shards this worker should handle
        n_shards = self.disk_dataset.get_number_shards()
        
        if worker_info is None:
            # Single-process data loading
            process_id = 0
            num_processes = 1
        else:
            # Multi-process data loading
            process_id = worker_info.id
            num_processes = worker_info.num_workers
            
        # Handle distributed training (DDP/FSDP)
        if dist.is_initialized():
            # Each GPU rank gets a portion of the data
            process_id += dist.get_rank() * num_processes
            num_processes *= dist.get_world_size()
            
        # Calculate which shards this worker should process
        # This ensures even distribution across all workers
        first_shard = (process_id * n_shards) // num_processes
        last_shard = ((process_id + 1) * n_shards) // num_processes
        
        # If this worker gets no shards, return empty iterator
        # This is safe because DataLoader will handle the coordination
        if first_shard >= last_shard:
            return iter([])
            
        shard_indices = list(range(first_shard, last_shard))
        
        # Iterate over epochs
        for epoch in range(self.epochs):
            # Shuffle shard order if not deterministic
            if not self.deterministic:
                np.random.shuffle(shard_indices)
                
            # Process each shard assigned to this worker
            for shard_idx in shard_indices:
                # Load the entire shard into memory
                # Note: This is synchronous - no thread pool complexity
                X, y, w, ids = self.disk_dataset.get_shard(shard_idx)
                
                # Handle empty shards gracefully
                if X.shape[0] == 0:
                    continue
                    
                # Shuffle samples within the shard if not deterministic
                n_samples = X.shape[0]
                if not self.deterministic:
                    # Create random permutation for this shard
                    perm = np.random.permutation(n_samples)
                else:
                    # Keep original order
                    perm = np.arange(n_samples)
                
                # Yield individual samples
                for i in perm:
                    # Extract individual sample
                    sample_x = X[i]
                    sample_y = y[i] if y is not None else None
                    sample_w = w[i] if w is not None else None
                    sample_id = ids[i]
                    
                    yield (sample_x, sample_y, sample_w, sample_id)


class _TorchImageDataset(torch.utils.data.IterableDataset):  # type: ignore

    def __init__(self,
                 image_dataset: ImageDataset,
                 epochs: int,
                 deterministic: bool,
                 batch_size: Optional[int] = None):
        """
        Parameters
        ----------
        image_dataset: ImageDataset
            The original ImageDataset which you want to convert to PyTorch Dataset
        epochs: int
            the number of times to iterate over the Dataset
        deterministic: bool
            if True, the data is produced in order.  If False, a different random
            permutation of the data is used for each epoch.
        batch_size: int
            the number of samples to return in each batch.  If None, each returned
            value is a single sample.
        """
        self.image_dataset = image_dataset
        self.epochs = epochs
        self.deterministic = deterministic
        self.batch_size = batch_size

    def __iter__(self):
        n_samples = self.image_dataset._X_shape[0]
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            first_sample = 0
            last_sample = n_samples
        else:
            first_sample = worker_info.id * n_samples // worker_info.num_workers
            last_sample = (worker_info.id +
                           1) * n_samples // worker_info.num_workers
        for epoch in range(self.epochs):
            if self.deterministic:
                order = first_sample + np.arange(last_sample - first_sample)
            else:
                # Ensure that every worker will pick the same random order for each epoch.
                random = np.random.RandomState(epoch)
                order = random.permutation(n_samples)[first_sample:last_sample]
            if self.batch_size is None:
                for i in order:
                    yield (self.image_dataset._get_image(
                        self.image_dataset._X, i),
                           self.image_dataset._get_image(
                               self.image_dataset._y, i),
                           self.image_dataset._w[i], self.image_dataset._ids[i])
            else:
                for i in range(0, len(order), self.batch_size):
                    indices = order[i:i + self.batch_size]
                    yield (self.image_dataset._get_image(
                        self.image_dataset._X, indices),
                           self.image_dataset._get_image(
                               self.image_dataset._y,
                               indices), self.image_dataset._w[indices],
                           self.image_dataset._ids[indices])

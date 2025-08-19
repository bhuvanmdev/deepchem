from typing import List, Tuple, Any
import deepchem as dc
import os
import glob
import logging
try:
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    pass

logger = logging.getLogger(__name__)

def collate_dataset_fn(batch_data: List[Tuple[Any, Any, Any, Any]], model):
    """Default Collate function for DeepChem datasets to work with PyTorch DataLoader.

    This function takes a batch of data from a PyTorch DataLoader and converts
    it into a format compatible with DeepChem models by wrapping it in a
    DeepChemBatch class that processes the data through the model's default
    generator and batch preparation methods.

    It does 3 important operations:
    1. Extracts the features (X), labels (Y), weights (W), and ids from the batch and arranges them correctly.
    2. Creates a NumpyDataset from these components and passes it to the model's default generator.
    3. Calls the model's `_prepare_batch` method that outputs the processed batch as a tuple of tensors.

    Parameters
    ----------
    batch: List[Tuple[Any, Any, Any, Any]]
        Batch of data from DataLoader containing tuples of (X, y, w, ids).
    model: TorchModel
        DeepChem model instance used for batch processing.

    Returns
    -------
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
        Processed batch tuple prepared by the model's _prepare_batch method.

    Examples
    --------
    >>> import deepchem as dc
    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>> from deepchem.data import _TorchIndexDiskDataset as TorchIndexDiskDataset
    >>> from deepchem.utils.lightning_utils import collate_dataset_fn
    >>>
    >>> # Load a dataset and create a model
    >>> tasks, datasets, _ = dc.molnet.load_clintox()
    >>> _, valid_dataset, _ = datasets
    >>> model = dc.models.MultitaskClassifier(
    ...     n_tasks=len(tasks),
    ...     n_features=1024,
    ...     layer_sizes=[1000],
    ...     device="cpu",
    ...     batch_size=16
    ... )
    >>>
    >>> # Create DataLoader with custom collate function
    >>> wrapped_dataset = TorchIndexDiskDataset(valid_dataset)
    >>> dataloader = DataLoader(
    ...     wrapped_dataset,
    ...     batch_size=16,
    ...     collate_fn=lambda batch: collate_dataset_fn(batch, model)
    ... )
    >>>
    >>> # Use in training loop
    >>> for batch in dataloader:
    ...     inputs, labels, weights = batch
    ...     # inputs, labels, weights are now properly formatted torch tensors
    ...     break
    """

    X, Y, W, ids = [], [], [], []
    X = [item[0] for item in batch_data]
    Y = [item[1] for item in batch_data]
    W = [item[2] for item in batch_data]
    ids = [item[3] for item in batch_data]
    processed_batch = next(
        iter(model.default_generator(dc.data.NumpyDataset(X, Y, W, ids))))
    return model._prepare_batch(processed_batch)


class RotatingModelCheckpoint(ModelCheckpoint):
    """Custom ModelCheckpoint that implements proper rotation for step-based checkpoints.
    
    This extends Lightning's ModelCheckpoint to provide automatic rotation of checkpoints
    based on creation time when using step-based saving without a monitor metric.
    """
    
    def __init__(self, *args, **kwargs):
        # Force save_top_k to -1 to save all checkpoints initially
        self.actual_save_top_k = kwargs.pop('save_top_k', 2)
        kwargs['save_top_k'] = -1  # Save all, we'll rotate manually
        kwargs['monitor'] = None   # No metric monitoring
        super().__init__(*args, **kwargs)
    
    def _save_checkpoint(self, trainer, filepath):
        """Override to implement rotation after saving."""
        # Save the checkpoint first
        super()._save_checkpoint(trainer, filepath)
        
        # Then clean up old checkpoints if needed
        self._rotate_checkpoints()
    
    def _rotate_checkpoints(self):
        """Remove old checkpoints to maintain save_top_k limit."""
        if self.actual_save_top_k <= 0 or not self.dirpath:
            return
            
        try:
            # Get all checkpoint files except last.ckpt
            checkpoint_pattern = os.path.join(str(self.dirpath), "*.ckpt")
            all_checkpoints = glob.glob(checkpoint_pattern)
            regular_checkpoints = [f for f in all_checkpoints 
                                 if not f.endswith("last.ckpt")]
            
            # Sort by modification time (newest first)
            regular_checkpoints.sort(key=os.path.getmtime, reverse=True)
            
            # Remove old checkpoints if we have more than actual_save_top_k
            if len(regular_checkpoints) > self.actual_save_top_k:
                checkpoints_to_remove = regular_checkpoints[self.actual_save_top_k:]
                for checkpoint_file in checkpoints_to_remove:
                    try:
                        os.remove(checkpoint_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")
        except Exception as e:
            logger.warning(f"Error during checkpoint rotation: {e}")
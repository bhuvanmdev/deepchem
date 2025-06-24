from torch.utils.data import DataLoader
import lightning as L
import deepchem as dc
from typing import Optional,Callable
from deepchem.models.lightning.utils import collate_dataset_wrapper, IndexDiskDatasetWrapper


class DeepChemLightningDataModule(L.LightningDataModule):
    """
    Lightning DataModule for DeepChem datasets.
    
    Args:
        dataset: DeepChem dataset for training
        batch_size: Batch size for training
        collate_fn: Custom collate function (default: collate_dataset_wrapper)
        num_workers: Number of workers for DataLoader
        model: DeepChem model for collate function
    """
    def __init__(
        self,
        dataset: dc.data.Dataset,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        num_workers: int = 0,
        model = None
    ):
        super().__init__()
        self._batch_size = batch_size
        self._dataset = IndexDiskDatasetWrapper(dataset)
        self._model = model
        
        if collate_fn is None and model is not None:
            self.collate_fn = lambda batch:collate_dataset_wrapper(batch=batch, model=model)
        else:
            self.collate_fn = collate_fn
            
        self.num_workers = num_workers

    def setup(self, stage: str):
        """Set up datasets for each stage."""
        if stage == "fit":
            self.train_dataset = self._dataset
        elif stage == "predict":
            self.predict_dataset = self._dataset

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """Return the prediction dataloader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
        )
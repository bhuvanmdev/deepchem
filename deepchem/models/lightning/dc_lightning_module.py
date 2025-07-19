import torch
import lightning as L  # noqa
from deepchem.models.torch_models import ModularTorchModel, TorchModel
import numpy as np
from deepchem.utils.typing import List, OneOrMany, Any, Tuple
from typing import Optional
from deepchem.trans import Transformer, undo_transforms
from deepchem.models.optimizers import LearningRateSchedule


class DCLightningModule(L.LightningModule):
    """DeepChem Lightning Module to be used with Lightning trainer.

    The lightning module is a wrapper over deepchem's torch model.
    This module directly works with pytorch lightning trainer
    which runs training for multiple epochs and also is responsible
    for setting up and training models on multiple GPUs.
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule

    Examples
    --------
    Training and prediction workflow with a GCN model:

    >>> import deepchem as dc
    >>> import lightning as L
    >>> from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule
    >>> from deepchem.models.lightning.dc_lightning_module import DCLightningModule
    >>> from deepchem.feat import MolGraphConvFeaturizer
    >>>
    >>> # Load and prepare dataset
    >>> tasks, dataset, transformers = dc.molnet.load_bace_classification(
    ...     featurizer=MolGraphConvFeaturizer(), reload=False)
    >>>
    >>> # Create a GCN model
    >>> model = dc.models.GCNModel(
    ...     mode='classification',
    ...     n_tasks=len(tasks),
    ...     number_atom_features=30,
    ...     batch_size=10,
    ...     learning_rate=0.0003
    ... )
    >>>
    >>> # Setup Lightning modules
    >>> data_module = DCLightningDatasetModule(
    ...     dataset=dataset[0],
    ...     batch_size=10,
    ...     model=model
    ... )
    >>> lightning_model = DCLightningModule(dc_model=model)
    >>>
    >>> # Setup trainer and fit
    >>> trainer = L.Trainer(
    ...     fast_dev_run=True,
    ...     accelerator="auto",
    ...     devices="auto",
    ...     logger=False,
    ...     enable_checkpointing=True
    ... )
    >>> # trainer.fit(model=lightning_model, datamodule=data_module)
    >>>
    >>> # Make predictions
    >>> # prediction_batches = trainer.predict(model=lightning_model, datamodule=data_module)

    Notes
    -----
    This class requires PyTorch to be installed.
    """

    def __init__(self, dc_model):
        """Create a new DCLightningModule.

        Parameters
        ----------
        dc_model: deepchem.models.torch_models.torch_model.TorchModel
            TorchModel to be wrapped inside the lightning module.
        """
        super().__init__()
        self.pt_model = dc_model.model
        self.model = self.pt_model
        self.dc_model = dc_model
        self.loss_mod = dc_model.loss
        self.optimizer = dc_model.optimizer
        self.output_types = dc_model.output_types
        self._prediction_outputs = dc_model._prediction_outputs
        self._loss_outputs = dc_model._loss_outputs
        self._variance_outputs = dc_model._variance_outputs
        self._other_outputs = dc_model._other_outputs
        self._loss_fn = dc_model._loss_fn
        self.uncertainty = getattr(dc_model, 'uncertainty', False)
        self.learning_rate = dc_model.learning_rate
        self._transformers: List[Transformer] = []
        self.other_output_types: Optional[OneOrMany[str]] = None
        self._built = True

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Union[torch.optim.Optimizer, List]
            PyTorch optimizer or list containing optimizer and scheduler.
        """
        self.dc_model._built = True
        py_optimizer = self.optimizer._create_pytorch_optimizer(
            self.pt_model.parameters())

        if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
            lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                py_optimizer)
            return [py_optimizer], [lr_schedule]

        return py_optimizer

    def training_step(self, batch, batch_idx):
        """Perform a training step.

        Parameters
        ----------
        batch: A tensor, tuple or list.
        batch_idx: Integer displaying index of this batch
        optimizer_idx: When using multiple optimizers, this argument will also be present.

        Returns
        -------
        loss_outputs: outputs of losses.
        """
        inputs, labels, weights = batch

        if isinstance(inputs, list):
            assert len(inputs) == 1
            inputs = inputs[0]

        if isinstance(self.dc_model, ModularTorchModel):
            loss = self.dc_model.loss_func(inputs, labels, weights)
        elif isinstance(self.dc_model, TorchModel):
            outputs = self.pt_model(inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]

            if self.dc_model._loss_outputs is not None:
                outputs = [outputs[i] for i in self.dc_model._loss_outputs]
            loss = self._loss_fn(outputs, labels, weights)

        self.log(
            "train_loss",
            loss.item(),
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
            batch_size=self.dc_model.batch_size,
        )

        return loss

    def predict_step(self, batch: Tuple[Any, Any, Any], batch_idx: int):
        """Perform a prediction step with optional support for uncertainty estimates and data transformations.

        This method was copied from TorchModel._predict and adapted for Lightning's predict_step interface.

        Parameters
        ----------
        batch: Tuple[Any, Any, Any]
            A tuple containing:
            - inputs: the input data for prediction,
            - labels: (unused in prediction, but maintained for consistency),
            - weights: (unused in prediction).
        batch_idx: int
            Index of the current batch.

        Returns
        -------
        Any
            Model predictions for this batch. Can be:
            - numpy array for single output models
            - list of numpy arrays for multi-output models
            - zip of (predictions, variances) if uncertainty is enabled
        """
        return TorchModel._predict(
            self=self,
            generator=iter(batch),
            uncertainty=self.uncertainty,
            transformers=self._transformers,
            other_output_types=self.other_output_types,
        )
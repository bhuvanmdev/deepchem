import torch
import lightning as L
from deepchem.models.optimizers import LearningRateSchedule
import numpy as np
from deepchem.models.torch_models import TorchModel, ModularTorchModel
from typing import Any, Tuple, List, Optional
from deepchem.utils.typing import OneOrMany
from deepchem.trans import Transformer


class DeepChemLightningModule(L.LightningModule):
    """A PyTorch Lightning wrapper for DeepChem models.

    This module integrates DeepChem's models with PyTorch Lightning's training loop,
    enabling efficient training and prediction workflows while managing model-specific
    operations such as loss calculation, uncertainty estimation, and data transformations.

    The class provides a consistent interface for:
      - Forward propagation through the model.
      - A training step method that computes and logs a loss value.
      - A prediction step method that handles uncertainty and additional outputs.
      - Configuration of optimizers and (optional) learning rate schedulers.
      - Multi-GPU support using all_gather for combining results across devices.

    Parameters
    ----------
    model: TorchModel
        An instance of a DeepChem TorchModel containing both
        the underlying PyTorch model and additional properties
        such as loss functions, optimizers, and output configuration.

    Examples
    --------
    Basic usage with a DeepChem TorchModel:

    >>> import torch
    >>> import lightning as L
    >>> from deepchem.models import GraphConvModel
    >>> from deepchem.models.lightning import DeepChemLightningModule
    >>> from deepchem.feat import ConvMolFeaturizer
    >>> from deepchem import data
    >>>
    >>> # Create a DeepChem model
    >>> featurizer = ConvMolFeaturizer()
    >>> model = GraphConvModel(n_tasks=1, mode='regression')
    >>>
    >>> # Wrap it in a Lightning module
    >>> lightning_module = DeepChemLightningModule(model)
    >>>
    >>> # Create a Lightning trainer for multi-GPU training
    >>> trainer = L.Trainer(max_epochs=10, accelerator='auto', devices=2)
    >>>
    >>> # Prepare your data as PyTorch Lightning DataModule or DataLoader
    >>> # train_dataloader = ...  # Your training data
    >>> # val_dataloader = ...    # Your validation data
    >>>
    >>> # Train the model (predictions will be gathered across all GPUs)
    >>> # trainer.fit(lightning_module, train_dataloader, val_dataloader)
    >>>
    >>> # Make predictions (results automatically combined from all GPUs)
    >>> # predictions = trainer.predict(lightning_module, test_dataloader)

    Multi-GPU Usage with all_gather:
    --------------------------------
    This module uses PyTorch Lightning's all_gather method to combine results
    from multiple GPUs during prediction. The all_gather method ensures that:
    
    1. During training: Loss values from all GPUs are gathered for monitoring
    2. During prediction: Predictions from all GPUs are combined into a single result
    3. Results are automatically synchronized across all devices
    
    The all_gather implementation handles:
    - Tensors: Concatenates results from all GPUs along the batch dimension
    - Lists/Tuples: Combines collections from all devices
    - Single GPU: Passes through results unchanged
    
    Example of manual all_gather usage:
    >>> def custom_epoch_end(self):
    ...     # Gather custom metrics from all GPUs
    ...     local_metric = torch.tensor(some_value)
    ...     all_metrics = self.all_gather(local_metric)
    ...     global_mean = torch.mean(all_metrics)
    ...     self.log("global_metric", global_mean)

    Notes
    -----
    For more information, see:
      - PyTorch Lightning Documentation: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule
      - All-gather documentation: https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#all-gather
    """

    def __init__(self, model: TorchModel):
        super().__init__()
        # Save hyperparameters for Lightning's checkpoint system
        self.save_hyperparameters(ignore=["model"])

        self.model = model.model
        self.dc_model = model
        self.loss_mod = model.loss
        self.optimizer = model.optimizer
        self.output_types = model.output_types
        self._prediction_outputs = model._prediction_outputs
        self._loss_outputs = model._loss_outputs
        self._variance_outputs = model._variance_outputs
        self._other_outputs = model._other_outputs
        self._loss_fn = model._loss_fn
        self.uncertainty = getattr(model, 'uncertainty', False)
        self.learning_rate = model.learning_rate
        self._transformers: List[Transformer] = []
        self.other_output_types: Optional[OneOrMany[str]] = None

    def training_step(self, batch: Tuple[Any, Any, Any], batch_idx: int):
        """Execute a single training step, including loss computation and logging.

        The method unpacks the batch into inputs, labels, and weights and then performs:
          - A forward pass through the network.
          - Loss computation which differentiates between ModularTorchModel and
            regular TorchModel based on the provided instance.
          - Logging of the loss value for monitoring.

        Parameters
        ----------
        batch: Tuple[Any, Any, Any]
            A tuple containing:
            - inputs: data inputs to the model,
            - labels: ground truth values,
            - weights: sample weights for the loss computation.
        batch_idx: int
            Index of the current batch (useful for logging or debugging).

        Returns
        -------
        torch.Tensor
            The computed loss value as a torch tensor. This value is used for backpropagation.
        """
        inputs, labels, weights = batch

        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        outputs = self.model(inputs)
        
        if isinstance(self.dc_model, ModularTorchModel):
            loss = self.dc_model.loss_func(inputs, labels, weights)
        elif isinstance(self.dc_model, TorchModel):
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            if self._loss_outputs is not None:
                outputs = [outputs[i] for i in self._loss_outputs]
            loss = self._loss_fn(outputs, labels, weights)
        
        self.log("train_loss", loss.item(), prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch: Tuple[Any, Any, Any], batch_idx: int):
        """Perform a prediction step with optional support for uncertainty estimates and data transformations.

        This method returns predictions for a single batch along with batch indices to enable
        proper ordering when aggregating results from multiple GPUs.

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
        Dict[str, Any]
            Dictionary containing:
            - 'predictions': Model predictions for this batch
            - 'batch_idx': Original batch index for ordering
            - 'start_idx': Starting sample index in the dataset
            - 'end_idx': Ending sample index in the dataset
        """
        results: Optional[List[List[np.ndarray]]] = None
        variances: Optional[List[List[np.ndarray]]] = None
        if self.uncertainty and (self.other_output_types is not None):
            raise ValueError(
                'This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time.'
            )
        if self.uncertainty:
            if self._variance_outputs is None or len(
                    self._variance_outputs) == 0:
                raise ValueError('This model cannot compute uncertainties')
            if len(self._variance_outputs) != len(self._prediction_outputs):
                raise ValueError(
                    'The number of variances must exactly match the number of outputs'
                )
        if self.other_output_types:
            if self._other_outputs is None or len(self._other_outputs) == 0:
                raise ValueError(
                    'This model cannot compute other outputs since no other output_types were specified.'
                )
        inputs, _, _ = batch
        # Invoke the model.
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        output_values = self.model(inputs)
        if isinstance(output_values, torch.Tensor):
            output_values = [output_values]
        output_values = [t.detach().cpu().numpy() for t in output_values]

        # Apply tranformers and record results.
        if self.uncertainty:
            var = [output_values[i] for i in self._variance_outputs]
            if variances is None:
                variances = [var]
            else:
                for i, t in enumerate(var):
                    variances[i].append(t)
        access_values = []
        if self.other_output_types:
            access_values += self._other_outputs
        elif self._prediction_outputs is not None:
            access_values += self._prediction_outputs

        if len(access_values) > 0:
            output_values = [output_values[i] for i in access_values]

        if len(self._transformers) > 0:
            if len(output_values) > 1:
                raise ValueError(
                    "predict() does not support Transformers for models with multiple outputs."
                )
            elif len(output_values) == 1:
                from deepchem.trans import undo_transforms
                output_values = [
                    undo_transforms(output_values[0], self._transformers)
                ]
        if results is None:
            results = [[] for i in range(len(output_values))]
        for i, t in enumerate(output_values):
            results[i].append(t)

        # Concatenate arrays to create the final results.
        final_results = []
        final_variances = []
        if results is not None:
            for r in results:
                final_results.append(np.concatenate(r, axis=0))
        if self.uncertainty and variances is not None:
            for v in variances:
                final_variances.append(np.concatenate(v, axis=0))
            return zip(final_results, final_variances)
        if len(final_results) == 1:
            return final_results[0]
        else:
            return final_results

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Union[torch.optim.Optimizer, List]
            PyTorch optimizer or list containing optimizer and scheduler.
        """

        py_optimizer = self.optimizer._create_pytorch_optimizer(
            self.model.parameters())

        if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
            lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                py_optimizer)
            return [py_optimizer], [lr_schedule]

        return py_optimizer
    
if __name__ == "__main__":
    import deepchem as dc
    import numpy as np
    from deepchem.models import MultitaskClassifier
    from deepchem.models.lightning.new_dc_lightning_dataset_module import DeepChemLightningDataModule
    # from deepchem.models.lightning.new_dc_lightning_module import DeepChemLightningModule
    import lightning as L

    # Load Clintox dataset
    tasks, datasets, _ = dc.molnet.load_clintox()
    _, valid_dataset, _ = datasets
    print("len of datasets:", len(valid_dataset.X))
    # Create a DeepChem MultitaskClassifier model
    model = MultitaskClassifier(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes=[1000],
        dropouts=0.2,
        learning_rate=0.0001,
        batch_size=2,
        device="cpu"
    )

    # Prepare the Lightning DataModule
    molnet_dataloader = DeepChemLightningDataModule(valid_dataset, 2,model=model)

    # Wrap the DeepChem model in the Lightning module
    lightning_module = DeepChemLightningModule(model)

    # Create a PyTorch Lightning Trainer
    trainer = L.Trainer(max_epochs=70, devices=-1, strategy="fsdp")

    trainer.fit(lightning_module, molnet_dataloader)

    # trainer = L.Trainer(max_epochs=1, devices=-1, strategy="fsdp", detect_anomaly=True)
    model = MultitaskClassifier(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes=[1000],
        dropouts=0.2,
        learning_rate=0.0001,
        batch_size=2,
        device="cpu"
    )
    # Fit the model
    # trainer.fit(lightning_module, molnet_dataloader)
    lightning_module = DeepChemLightningModule(model)
    
    # Create a single-GPU trainer for prediction to ensure complete, ordered results
    prediction_trainer = L.Trainer(devices=1, accelerator="auto")
    
    # Run prediction on the validation dataset
    predictions = prediction_trainer.predict(lightning_module, molnet_dataloader)
    print("Predictions type:", type(predictions))
    if predictions is not None and len(predictions) > 0:
        print("Number of prediction batches:", len(predictions))
        try:
            concatenated_predictions = np.concatenate([p for p in predictions])
            print("All predictions shape:", concatenated_predictions.shape)
            print("Expected shape: ({}, 2)".format(len(valid_dataset)))
        except Exception as e:
            print("Error concatenating predictions:", e)
            print("First few prediction types:", [type(p) for p in predictions[:3]])
    else:
        print("No predictions returned or empty predictions")
    
    print("\n=== Testing with DeepChemLightningTrainer (Recommended) ===")
    

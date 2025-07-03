from deepchem.models.lightning.new_dc_lightning_dataset_module import DeepChemLightningDataModule
from deepchem.models.lightning.new_dc_lightning_module import DeepChemLightningModule
from deepchem.models.torch_models import TorchModel
from rdkit import rdBase
import deepchem as dc
from deepchem.data import Dataset
import lightning as L
from typing import List, Optional, Union, Dict, Tuple, Any
from deepchem.utils.typing import OneOrMany
from deepchem.trans import Transformer
from deepchem.metrics import Metric
from deepchem.utils.evaluate import _process_metric_input
import numpy as np
import logging

rdBase.DisableLog('rdApp.warning')
logger = logging.getLogger(__name__)

# Type aliases for compatibility with evaluate.py
Score = Dict[str, float]
Metrics = Union[Metric, Any, List[Metric], List[Any]]


class DeepChemLightningTrainer:
    """A wrapper class that handles the training and inference of DeepChem models using Lightning.

    This class provides a high-level interface for training and running inference
    on DeepChem models using PyTorch Lightning's training infrastructure. It wraps
    DeepChem models in Lightning modules and handles data loading, training loops,
    and checkpoint management.

    Parameters
    ----------
    model: TorchModel
        Initialized DeepChem model to be trained or used for inference.
    batch_size: int, default 32
        Batch size for training and prediction data loaders.
    **trainer_kwargs
        Additional keyword arguments passed to the Lightning Trainer.

    Examples
    --------
    >>> import deepchem as dc
    >>> import lightning as L
    >>> from deepchem.models.lightning.trainer2 import DeepChemLightningTrainer
    >>> tasks, datasets, _ = dc.molnet.load_clintox()
    >>> _, valid_dataset, _ = datasets
    >>> model = dc.models.MultitaskClassifier(
    ...     n_tasks=len(tasks),
    ...     n_features=1024,
    ...     layer_sizes=[1000],
    ...     dropouts=0.2,
    ...     learning_rate=0.0001,
    ...     device="cpu",
    ...     batch_size=16
    ... )
    >>> trainer = DeepChemLightningTrainer(
    ...     model=model,
    ...     batch_size=16,
    ...     max_epochs=30,
    ...     accelerator="cpu",
    ...     log_every_n_steps=1,
    ...     fast_dev_run=True
    ... )
    >>> trainer.fit(valid_dataset)
    >>> predictions = trainer.predict(valid_dataset)
    >>> trainer.save_checkpoint("model.ckpt")
    >>> # To reload:
    >>> trainer2 = DeepChemLightningTrainer.load_checkpoint("model.ckpt", model=model)
    """

    def __init__(self,
                 model: TorchModel,
                 batch_size: int = 32,
                 **trainer_kwargs):
        self.model = model
        self.batch_size = batch_size
        self.trainer_kwargs = trainer_kwargs

        # Set default trainer arguments if not provided
        if 'accelerator' not in trainer_kwargs:
            self.trainer_kwargs['accelerator'] = 'auto'
        if 'devices' not in trainer_kwargs:
            self.trainer_kwargs['devices'] = 'auto'
        if 'strategy' not in trainer_kwargs:
            self.trainer_kwargs['strategy'] = 'auto'
        if 'max_epochs' not in trainer_kwargs:
            self.trainer_kwargs['max_epochs'] = 100

        # TODO: set up logger if not provided

        # Create the Lightning module
        self.lightning_model = DeepChemLightningModule(model)

    def fit(self, train_dataset: Dataset, num_workers: int = 4):
        """Train the model on the provided dataset.

        Parameters
        ----------
        train_dataset: dc.data.Dataset
            DeepChem dataset for training.
        num_workers: int, default 4
            Number of workers for DataLoader.

        Returns
        -------
        None
            The trainer object is modified in place after fitting.
        """
        # Set log_every_n_steps if not provided
        if 'log_every_n_steps' not in self.trainer_kwargs:
            dataset_size = len(train_dataset)
            self.trainer_kwargs['log_every_n_steps'] = max(
                1, dataset_size // (int(self.batch_size) * 2))

        self.lightning_model = self.lightning_model.train()

        # Create data module
        data_module = DeepChemLightningDataModule(dataset=train_dataset,
                                                  batch_size=int(self.batch_size),
                                                  num_workers=num_workers,
                                                  model=self.model)

        # Create trainer
        self.trainer = L.Trainer(**self.trainer_kwargs)

        # Train the model
        self.trainer.fit(self.lightning_model, data_module)

    def predict(self,
                dataset: Dataset,
                transformers: List[Transformer] = [],
                other_output_types: Optional[OneOrMany[str]] = None,
                num_workers: int = 0,
                uncertainty: Optional[bool] = None,
                ckpt_path: Optional[str] = None,
                use_multi_gpu: bool = False):
        """Run inference on the provided dataset.

        Parameters
        ----------
        dataset: dc.data.Dataset
            DeepChem dataset for prediction.
        transformers: List[Transformer], default []
            List of transformers to apply to predictions.
        other_output_types: Optional[OneOrMany[str]], default None
            List of other output types to compute.
        num_workers: int, default 4
            Number of workers for DataLoader.
        uncertainty: Optional[bool], default None
            Whether to compute uncertainty estimates.
        ckpt_path: Optional[str], default None
            Path to a checkpoint file to load model weights from.
        use_multi_gpu: bool, default False
            Whether to use multi-GPU prediction. If False, forces single-GPU for correct ordering.

        Returns
        -------
        List
            Predictions from the model.
        """
        
        if not use_multi_gpu:
            # Default behavior: force single-GPU prediction for correct ordering
            predict_trainer_kwargs = self.trainer_kwargs.copy()
            predict_trainer_kwargs['devices'] = 1  # Force single GPU
            predict_trainer_kwargs['strategy'] = 'auto'  # Use simple strategy for prediction
            print("[WARNING] Using single GPU for prediction to ensure correct sample ordering")
            print("[INFO] Set use_multi_gpu=True to enable experimental multi-GPU prediction")
            self.trainer = L.Trainer(**predict_trainer_kwargs)
        else:
            # Multi-GPU prediction - use DDP instead of FSDP for compatibility
            predict_trainer_kwargs = self.trainer_kwargs.copy()
            if predict_trainer_kwargs.get('strategy') == 'fsdp':
                predict_trainer_kwargs['strategy'] = 'ddp'  # Use DDP for prediction instead of FSDP
                print("[INFO] Switching from FSDP to DDP strategy for multi-GPU prediction")
            self.trainer = L.Trainer(**predict_trainer_kwargs)

        # Create data module
        data_module = DeepChemLightningDataModule(dataset=dataset,
                                                  batch_size=int(self.batch_size),
                                                  num_workers=num_workers,
                                                  model=self.model)
        self.lightning_model = self.lightning_model.eval()
        
        # Set prediction parameters
        self.lightning_model._transformers = transformers
        self.lightning_model.other_output_types = other_output_types
        if uncertainty is not None:
            self.lightning_model.uncertainty = uncertainty

        # Run prediction
        predictions = self.trainer.predict(self.lightning_model,
                                           datamodule=data_module,
                                           return_predictions=True,
                                           ckpt_path=ckpt_path)



        return predictions

    def evaluate(self,
                 dataset: Dataset,
                 metrics: Metrics,
                 transformers: List[Transformer] = [],
                 per_task_metrics: bool = False,
                 use_sample_weights: bool = False,
                 n_classes: int = 2,
                 num_workers: int = 0) -> Union[Score, Tuple[Score, Score]]:
        """
        Evaluate model performance on a dataset using Lightning for multi-GPU support.
        
        This method provides a Lightning-compatible version of the standard Evaluator
        functionality, enabling distributed evaluation across multiple GPUs.
        
        Parameters
        ----------
        dataset: Dataset
            DeepChem dataset to evaluate on.
        metrics: Metrics
            The set of metrics to compute. Can be a single metric, list of metrics,
            or metric functions.
        transformers: List[Transformer], default []
            List of transformers that were applied to the dataset.
        per_task_metrics: bool, default False
            If true, return computed metric for each task on multitask dataset.
        use_sample_weights: bool, default False
            If set, use per-sample weights.
        n_classes: int, default 2
            Number of unique classes for classification metrics.
        num_workers: int, default 0
            Number of workers for DataLoader.
            
        Returns
        -------
        Union[Score, Tuple[Score, Score]]
            Dictionary mapping metric names to scores. If per_task_metrics is True,
            returns a tuple of (multitask_scores, all_task_scores).
        """
        # Process input metrics
        processed_metrics = _process_metric_input(metrics)
        
        # Get predictions using Lightning's multi-GPU predict
        y_pred = self.predict(dataset, transformers=transformers, num_workers=num_workers)
        
        # Debug: Print prediction structure to understand multi-GPU output
        print(f"Debug: y_pred type: {type(y_pred)}")
        if isinstance(y_pred, list):
            print(f"Debug: y_pred length: {len(y_pred)}")
            for i, item in enumerate(y_pred):
                print(f"Debug: y_pred[{i}] type: {type(item)}, shape: {getattr(item, 'shape', 'no shape')}")
        
        # Handle multi-GPU prediction concatenation robustly
        if isinstance(y_pred, list) and len(y_pred) > 0:
            # First, collect all prediction arrays
            all_predictions = []
            
            def collect_arrays(obj):
                """Recursively collect numpy arrays from nested structures."""
                if isinstance(obj, np.ndarray):
                    all_predictions.append(obj)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        collect_arrays(item)
                elif obj is not None:
                    # Convert other types to numpy arrays
                    try:
                        arr = np.array(obj)
                        if arr.size > 0:
                            all_predictions.append(arr)
                    except:
                        pass
            
            # Collect all arrays recursively
            collect_arrays(y_pred)
            
            # Debug: Print collected arrays info
            print(f"Debug: Collected {len(all_predictions)} prediction arrays")
            for i, arr in enumerate(all_predictions):
                print(f"Debug: Array {i} shape: {arr.shape}")
            
            # Concatenate all collected arrays
            if all_predictions:
                y_pred = np.concatenate(all_predictions, axis=0)
            else:
                y_pred = np.array([])
        else:
            # Ensure y_pred is a numpy array
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred) if y_pred is not None else np.array([])
        
        # Get true labels and weights
        y = dataset.y
        w = dataset.w
        
        # Debug: Print final shapes
        print(f"Debug: Final y_pred shape: {y_pred.shape}")
        print(f"Debug: Dataset y shape: {y.shape}")
        print(f"Debug: Dataset w shape: {w.shape}")
        
        # Ensure predictions match dataset size - trim if necessary due to multi-GPU padding
        if len(y_pred) > len(y):
            print(f"Debug: Trimming predictions from {len(y_pred)} to {len(y)}")
            y_pred = y_pred[:len(y)]
        
        # Apply transformers to true labels (undo transforms)
        output_transformers = [t for t in transformers if t.transform_y]
        if output_transformers:
            import deepchem.trans
            y = deepchem.trans.undo_transforms(y, output_transformers)
        
        n_tasks = len(dataset.get_task_names())
        
        multitask_scores = {}
        all_task_scores = {}
        
        # Compute metrics
        for metric in processed_metrics:
            results = metric.compute_metric(
                y,
                y_pred,
                w,
                per_task_metrics=per_task_metrics,
                n_tasks=n_tasks,
                n_classes=n_classes,
                use_sample_weights=use_sample_weights
            )
            
            if per_task_metrics:
                multitask_scores[metric.name], computed_metrics = results
                all_task_scores[metric.name] = computed_metrics
            else:
                multitask_scores[metric.name] = results
        
        if not per_task_metrics:
            return multitask_scores
        else:
            return multitask_scores, all_task_scores

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint using Lightning's native checkpointing.

        Parameters
        ----------
        filepath: str
            Path to save the checkpoint file.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not hasattr(self, 'trainer'):
            raise ValueError(
                "Model has not been trained yet. Please call fit() first.")
        
        self.trainer.save_checkpoint(filepath)

    def restore(self, checkpoint_path: str):
        """Restore model from checkpoint using Lightning's native loading.

        Parameters
        ----------
        checkpoint_path: str
            Path to the checkpoint file.
        """
        # Load Lightning module with all hyperparameters and state
        self.lightning_model = DeepChemLightningModule.load_from_checkpoint(
            checkpoint_path, model=self.model)
        # Keep the original batch_size from trainer initialization

    def resume_training(self, 
                       train_dataset, 
                       checkpoint_path: str,
                       num_workers: int = 4):
        """Resume training from a checkpoint.

        Parameters
        ----------
        train_dataset: Dataset
            Training dataset.
        checkpoint_path: str
            Path to checkpoint to resume from.
        num_workers: int, default 4
            Number of workers for DataLoader.
        """
        # Set log_every_n_steps if not provided
        if 'log_every_n_steps' not in self.trainer_kwargs:
            dataset_size = len(train_dataset)
            self.trainer_kwargs['log_every_n_steps'] = max(
                1, dataset_size // (int(self.batch_size) * 2))

        # Create data module
        data_module = DeepChemLightningDataModule(dataset=train_dataset,
                                                  batch_size=int(self.batch_size),
                                                  num_workers=num_workers,
                                                  model=self.model)

        # Create trainer
        self.trainer = L.Trainer(**self.trainer_kwargs)

        # Resume training from checkpoint
        self.trainer.fit(self.lightning_model, data_module, ckpt_path=checkpoint_path)

    @staticmethod
    def load_checkpoint(filepath: str,
                        model: TorchModel,
                        batch_size: int = 32,
                        **trainer_kwargs):
        """Load model from checkpoint and create a new trainer instance.

        Parameters
        ----------
        filepath: str
            Path to checkpoint file (.ckpt).
        model: TorchModel
            DeepChem model instance to load weights into.
        batch_size: int, default 32
            Batch size for the trainer/model.
        **trainer_kwargs
            Additional trainer arguments.

        Returns
        -------
        DeepChemLightningTrainer
            New trainer instance with loaded model.
        """
        # Create trainer first
        trainer = DeepChemLightningTrainer(model=model,
                                           batch_size=batch_size,
                                           **trainer_kwargs)
        
        # Load the checkpoint
        trainer.restore(filepath)
        
        return trainer

    def _post_process_multi_gpu_predictions(self, predictions: List, expected_size: int):
        """
        Post-process multi-GPU predictions to handle potential ordering issues.
        
        This method attempts to clean up multi-GPU prediction results by:
        1. Flattening nested prediction structures
        2. Trimming to expected dataset size
        3. Removing potential duplicates from DDP padding
        
        Parameters
        ----------
        predictions: List
            Raw predictions from multi-GPU trainer.predict()
        expected_size: int
            Expected number of samples in the dataset
            
        Returns
        -------
        np.ndarray or List
            Cleaned predictions
        """
        print(f"[DEBUG] Post-processing multi-GPU predictions")
        print(f"[DEBUG] Raw predictions type: {type(predictions)}, length: {len(predictions) if hasattr(predictions, '__len__') else 'N/A'}")
        
        # Collect all prediction arrays
        all_predictions = []
        
        def collect_arrays(obj):
            """Recursively collect numpy arrays from nested structures."""
            if isinstance(obj, np.ndarray):
                all_predictions.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    collect_arrays(item)
            elif obj is not None:
                # Convert other types to numpy arrays
                try:
                    arr = np.array(obj)
                    if arr.size > 0:
                        all_predictions.append(arr)
                except:
                    pass
        
        # Collect all arrays recursively
        collect_arrays(predictions)
        
        print(f"[DEBUG] Collected {len(all_predictions)} prediction arrays")
        for i, arr in enumerate(all_predictions):
            print(f"[DEBUG] Array {i} shape: {arr.shape}")
        
        # Concatenate and trim to expected size
        if all_predictions:
            concatenated = np.concatenate(all_predictions, axis=0)
            
            # Trim to expected size (removes duplicates from DDP padding)
            if len(concatenated) > expected_size:
                print(f"[DEBUG] Trimming predictions from {len(concatenated)} to {expected_size}")
                concatenated = concatenated[:expected_size]
            
            print(f"[DEBUG] Final prediction shape: {concatenated.shape}")
            return concatenated
        else:
            print("[WARNING] No predictions found in multi-GPU output")
            return np.array([])

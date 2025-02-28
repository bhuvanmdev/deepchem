from typing import Union, List

# import deepchem as dc


# class DistributedTrainer():
#     r"""Perform distributed training of DeepChem models

#     DistributedTrainer provides an interface for scaling the training of DeepChem
#     model to multiple GPUs and nodes. To achieve this, it uses
#     `Lightning <https://lightning.ai/>`_ under the hood. A DeepChem model
#     is converted to a `LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`_
#     using :class:`~deepchem.models.lightning.DCLightningModule` and a DeepChem dataset
#     is converted to a PyTorch iterable dataset using :class:`~deepchem.models.lightning.DCLightningDatasetModule`.
#     The model and dataset are then trained using Lightning `Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

#     Example
#     -------
#     .. code-block:: python

#         import deepchem as dc
#         from deepchem.models.trainer import DistributedTrainer

#         dataset = dc.data.DiskDataset('zinc100k')

#         atom_vocab = GroverAtomVocabularyBuilder.load('zinc100k_atom_vocab.json')
#         bond_vocab = GroverBondVocabularyBuilder.load('zinc100k_bond_vocab.json')

#         model = GroverModel(task='pretraining',
#                             mode='pretraining',
#                             node_fdim=151,
#                             edge_fdim=165,
#                             features_dim=2048,
#                             functional_group_size=85,
#                             hidden_size=128,
#                             learning_rate=0.0001,
#                             batch_size=1,
#                             dropout=0.1,
#                             ffn_num_layers=2,
#                             ffn_hidden_size=5,
#                             num_attn_heads=8,
#                             attn_hidden_size=128,
#                             atom_vocab=atom_vocab,
#                             bond_vocab=bond_vocab)

#         trainer = DistributedTrainer(max_epochs=1,
#                                      batch_size=64,
#                                      num_workers=0,
#                                      accelerator='gpu',
#                                      distributed_strategy='ddp')
#         trainer.fit(model, dataset)

#     """

#     def __init__(self,
#                  max_epochs: int,
#                  batch_size: int,
#                  devices: Union[str, int, List[int]] = 'auto',
#                  accelerator: str = 'auto',
#                  distributed_strategy: str = 'auto'):
#         """Initialise DistributedTrainer

#         Parameters
#         ----------
#         max_epochs: int
#             Maximum number of training epochs
#         batch_size: int
#             Batch size
#         devices: str, optional (default: 'auto')
#             Number of devices to train on (int) or which devices to train on (list or str).
#         accelerator: str, optional (default: 'auto')
#             Specify the accelerator to train the models (cpu, gpu, mps)
#         distributed_strategy: str, (default: 'auto')
#             Specify training strategy (ddp - distributed data parallel, fsdp - fully sharded data parallel, etc).
#         When strategy is auto, the Trainer chooses the default strategy for the hardware (ddp for torch). Also, refer to Lightning's docs on strategy `here <https://lightning.ai/docs/pytorch/stable/extensions/strategy.html>`_ .
#         """
#         self.max_epochs = max_epochs
#         self.batch_size = batch_size
#         self.devices = devices
#         self.accelerator = accelerator
#         self.distributed_strategy = distributed_strategydc_

#     def fit(self, model: 'dc.models.Model', dataset: 'dc.data.Dataset') -> None:
#         """Train the model using DistributedTrainer

#         Parameters
#         ----------
#         model: dc.models.Model
#             A deepchem model to train
#         dataset: dc.data.DiskDataset
#             A deepchem Dataset for training
#         """
#         import lightning as L
#         from deepchem.models.lightning import DCLightningModule, DCLightningDatasetModule
#         lit_model = DCLightningModule(model)
#         lit_dataset = DCLightningDatasetModule(dataset,
#                                                batch_size=self.batch_size)
#         trainer = L.Trainer(max_epochs=self.max_epochs,
#                             devices=self.devices,
#                             accelerator=self.accelerator)
#         trainer.fit(lit_model, lit_dataset)
#         return

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import numpy as np

# Assuming your NumpyDataset looks something like this
class NumpyDataset(Dataset):
    def __init__(self, X, y, w=None, ids=None):
        self.X = X
        self.y = y
        self.w = w if w is not None else np.ones(len(X))
        self.ids = ids if ids is not None else np.arange(len(X))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': torch.tensor(self.X[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.float32),
            'w': torch.tensor(self.w[idx], dtype=torch.float32),
            'ids': self.ids[idx]
        }

# Dataset container for splitting into train/val/test
class DatasetContainer:
    def __init__(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Calculate sizes
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
    
    @classmethod
    def from_numpy(cls, X, y, w=None, ids=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Create a DatasetContainer directly from numpy arrays"""
        dataset = NumpyDataset(X, y, w, ids)
        return cls(dataset, train_ratio, val_ratio, test_ratio, seed)

# Lightning wrapper for your TorchModel
class LightningModelWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Adapt to your specific loss calculation
        x, y, w = batch['X'], batch['y'], batch['w']
        y_pred = self.model(x)
        loss = self.model.compute_loss(y_pred, y, w)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, w = batch['X'], batch['y'], batch['w']
        y_pred = self.model(x)
        loss = self.model.compute_loss(y_pred, y, w)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, w = batch['X'], batch['y'], batch['w']
        y_pred = self.model(x)
        loss = self.model.compute_loss(y_pred, y, w)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x = batch['X']
        return self.model.predict_on_batch(x)
    
    def configure_optimizers(self):
        # Use your model's optimizer settings or pass them as params
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    # Proxy methods to access original TorchModel methods
    def predict_on_generator(self, data_gen, **kwargs):
        return self.model.predict_on_generator(data_gen, **kwargs)
    
    def predict_on_batch(self, batch, **kwargs):
        return self.model.predict_on_batch(batch, **kwargs)
    
    def predict_uncertainty_on_batch(self, batch, **kwargs):
        return self.model.predict_uncertainty_on_batch(batch, **kwargs)
    
    def predict_embedding(self, data, **kwargs):
        return self.model.predict_embedding(data, **kwargs)
    
    def predict_uncertainty(self, data, **kwargs):
        return self.model.predict_uncertainty(data, **kwargs)
    
    def evaluate_generator(self, data_gen, **kwargs):
        return self.model.evaluate_generator(data_gen, **kwargs)
    
    def compute_saliency(self, data, **kwargs):
        return self.model.compute_saliency(data, **kwargs)
    
    def _prepare_batch(self, batch, **kwargs):
        return self.model._prepare_batch(batch, **kwargs)
    
    def default_generator(self, data, **kwargs):
        return self.model.default_generator(data, **kwargs)

# Lightning Data Module for your dataset
class TorchModelDataModule(pl.LightningDataModule):
    def __init__(self, dataset_container, batch_size=32, num_workers=0):
        super().__init__()
        self.dataset_container = dataset_container
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset_container.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset_container.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.dataset_container.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

# Main FSDP wrapper class
class FSDPWrapper:
    def __init__(self, 
                 precision="bf16-mixed", 
                 min_num_params=1e6,
                 cpu_offload=False, 
                 use_gloo=True):
        """
        FSDP Wrapper for TorchModel
        
        Args:
            precision: Precision to use for training ("bf16-mixed", "fp16-mixed", "32")
            min_num_params: Minimum number of parameters for auto-wrapping
            cpu_offload: Whether to offload parameters to CPU
            use_gloo: Whether to use gloo backend (for Windows) instead of nccl
        """
        self.precision = precision
        self.min_num_params = min_num_params
        self.cpu_offload = cpu_offload
        self.use_gloo = use_gloo
        
        # Set environment variables for gloo backend
        if use_gloo:
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    
    def _get_fsdp_strategy(self):
        """Configure FSDP strategy based on settings"""
        # Set up mixed precision policy if needed
        mp_policy = None
        if self.precision == "bf16-mixed":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self.precision == "fp16-mixed":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        
        # Create auto-wrap policy based on parameter size
        def auto_wrap_policy(module):
            return sum(p.numel() for p in module.parameters()) >= self.min_num_params
        
        # Return configured FSDP strategy
        return FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            activation_checkpointing=None,  # Set to your transformer blocks if needed
            sharding_strategy="FULL_SHARD",  # Options: SHARD_GRAD_OP, HYBRID_SHARD
            cpu_offload=self.cpu_offload,
            limit_all_gathers=True,
            state_dict_type="full",  # For easier checkpointing
        )
    
    def wrap(self, torch_model):
        """Wrap a TorchModel with Lightning and FSDP"""
        return LightningModelWrapper(torch_model)
    
    def train(self, 
              torch_model, 
              dataset_container, 
              batch_size=32, 
              max_epochs=10, 
              num_gpus=1, 
              log_dir="lightning_logs",
              ckpt_dir="checkpoints",
              num_workers=0):
        """
        Train a model using FSDP
        
        Args:
            torch_model: Your TorchModel instance
            dataset_container: DatasetContainer with your data
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs to train
            num_gpus: Number of GPUs to use
            log_dir: Directory for logs
            ckpt_dir: Directory for checkpoints
            num_workers: Number of workers for data loading
        
        Returns:
            tuple: (trainer, wrapped_model)
        """
        # Wrap the model with Lightning
        lightning_model = self.wrap(torch_model)
        
        # Create data module
        data_module = TorchModelDataModule(
            dataset_container=dataset_container,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # Get FSDP strategy
        fsdp_strategy = self._get_fsdp_strategy()
        
        # Set up appropriate precision for the trainer
        if self.precision == "bf16-mixed":
            precision = "bf16-mixed"
        elif self.precision == "fp16-mixed":
            precision = "16-mixed"
        else:
            precision = "32-true"
        
        # Create Lightning trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            strategy=fsdp_strategy,
            accelerator="gpu" if num_gpus > 0 else "cpu",
            devices=num_gpus if num_gpus > 0 else None,
            precision=precision,
            # logger=pl.loggers.TensorBoardLogger(log_dir),
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename='{epoch}-{val_loss:.2f}',
                    save_top_k=3,
                    monitor='val_loss'
                ),
                pl.callbacks.LearningRateMonitor(),
            ]
        )
        
        # Train the model
        trainer.fit(lightning_model, datamodule=data_module)
        
        return trainer, lightning_model
    
    def load_model(self, checkpoint_path, torch_model_class, model_args=None):
        """
        Load a model from a checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint
            torch_model_class: Your TorchModel class
            model_args: Arguments to pass to the model constructor
            
        Returns:
            LightningModelWrapper: Loaded model
        """
        # Initialize model if args provided
        if model_args is not None:
            base_model = torch_model_class(**model_args)
            wrapped_model = self.wrap(base_model)
            
            # Load weights
            wrapped_model = LightningModelWrapper.load_from_checkpoint(
                checkpoint_path,
                model=base_model
            )
        else:
            # Load directly
            wrapped_model = LightningModelWrapper.load_from_checkpoint(checkpoint_path)
        
        return wrapped_model

# # Example usage function
# def example_usage():
#     """Example of how to use the FSDP wrapper"""
#     # Import your TorchModel
#     from your_model_module import YourTorchModel
    
#     # 1. Create your model normally
#     model = YourTorchModel(input_dim=128, hidden_dim=256)
    
#     # 2. Create dataset (from your NumpyDataset)
#     X = np.random.randn(1000, 128)
#     y = np.random.randn(1000, 1)
#     w = np.ones(1000)
#     ids = np.arange(1000)
    
#     # Create DatasetContainer
#     dataset_container = DatasetContainer.from_numpy(
#         X=X, y=y, w=w, ids=ids,
#         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
#     )
    
#     # 3. Create FSDP wrapper
#     fsdp_wrapper = FSDPWrapper(
#         precision="fp16-mixed",  # Could be "bf16-mixed" or "32"
#         use_gloo=True  # For Windows
#     )
    
#     # 4. Train with FSDP
#     trainer, wrapped_model = fsdp_wrapper.train(
#         torch_model=model,
#         dataset_container=dataset_container,
#         batch_size=64,
#         max_epochs=20,
#         num_gpus=1,  # Adjust based on your hardware
#         log_dir="lightning_logs",
#         ckpt_dir="checkpoints",
#         num_workers=0  # Important for Windows compatibility
#     )
    
#     # 5. Use the trained model
#     # The wrapped_model has all the methods of your original TorchModel
#     test_data = np.random.randn(10, 128)
#     predictions = wrapped_model.predict_on_batch(torch.tensor(test_data, dtype=torch.float32))
    
#     # You can also compute uncertainty if your model supports it
#     uncertainty = wrapped_model.predict_uncertainty(test_data)
    
#     # 6. Save and load
#     trainer.save_checkpoint("final_model.ckpt")
    
#     # Later, load the model
#     loaded_model = fsdp_wrapper.load_model(
#         checkpoint_path="final_model.ckpt",
#         torch_model_class=YourTorchModel,
#         model_args={'input_dim': 128, 'hidden_dim': 256}
#     )
    
#     return trainer, wrapped_model
    

import numpy as np
import pytest
from flaky import flaky

from deepchem.data import NumpyDataset
from deepchem.metrics import Metric, roc_auc_score, mean_absolute_error
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.feat import WeaveFeaturizer
def get_dataset(mode='classification',
                featurizer='GraphConv',
                num_tasks=2,
                data_points=20):
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(
            featurizer, reload=False)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer,
                                                        reload=False)

    train, _, _ = all_dataset
    for _ in range(1, num_tasks):
        tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, len(tasks)))
        metric = Metric(roc_auc_score, np.mean, mode="classification")
    else:
        y = np.random.normal(size=(data_points, len(tasks)))
        metric = Metric(mean_absolute_error, mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric
def main():
    # 1. Initialize your model normally, without any FSDP-specific code
    from deepchem.models.torch_models import WeaveModel
    from deepchem.molnet import load_bace_classification, load_delaney
    model = WeaveModel(1)
    
    # 2. Prepare your data (using your existing data preparation methods)
    tasks,(train_data,test_data,val_data),_ = load_bace_classification()
    
    # 3. Create the FSDP wrapper
    # If your model has transformer layers, specify them for better FSDP wrapping
    fsdp_wrapper = FSDPWrapper(
        model=model,
        precision="bf16-mixed",  # Use bfloat16 mixed precision (requires modern GPUs)
        transformer_layer_cls=None,#[YourCustomModel.TransformerBlock],  # Specify transformer layer classes if you have them
        batch_size=64,
        num_gpus=-1,  # Use 4 GPUs, or -1 for all available
        max_epochs=20,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={'lr': 2e-4, 'weight_decay': 0.01},
        # Optional scheduler
        scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs={'T_max': 20}
    )
    
    # 4. Train the model with FSDP
    trainer, lightning_model = fsdp_wrapper.train(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data
    )
    
    # 5. Save the trained model
    fsdp_wrapper.save_checkpoint("my_model_fsdp_trained.pt")
    
    # 6. Now you can use all your original methods from TorchModel
    # These calls will be delegated to your original model
    
    # Example: Use predict_on_batch
    # test_batch = get_test_batch()
    # predictions = fsdp_wrapper.predict_on_batch(test_batch)
    
    # Example: Use predict_uncertainty
    uncertainty_results = fsdp_wrapper.predict_uncertainty(test_data)
    
    # Example: Use compute_saliency
    # saliency_map = fsdp_wrapper.compute_saliency(test_batch)
    
    # Example: Use predict_embedding
    embeddings = fsdp_wrapper.predict_embedding(test_data)
    
    # Example: Use evaluate_generator
    eval_results = fsdp_wrapper.evaluate_generator(
        fsdp_wrapper.model.default_generator(test_data, batch_size=32)
    )
    
    print("Training completed and model evaluated successfully!")



# # Example usage function
def example_usage():
    """Example of how to use the FSDP wrapper"""
    # Import your TorchModel
    from deepchem.models.torch_models import WeaveModel
    
    # 1. Create your model normally
    model = WeaveModel(1)
    
    # 2. Create dataset (from your NumpyDataset)
    X = np.random.randn(1000, 75)
    y = np.random.randn(1000, 1)
    w = np.ones(1000)
    ids = np.arange(1000)
    
    # Create DatasetContainer
    dataset_container = DatasetContainer.from_numpy(
        X=X, y=y, w=w, ids=ids,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # 3. Create FSDP wrapper
    fsdp_wrapper = FSDPWrapper(
        precision="fp16-mixed",  # Could be "bf16-mixed" or "32"
        use_gloo=True  # For Windows
    )
    
    # 4. Train with FSDP
    trainer, wrapped_model = fsdp_wrapper.train(
        torch_model=model,
        dataset_container=dataset_container,
        batch_size=64,
        max_epochs=20,
        num_gpus=1,  # Adjust based on your hardware
        log_dir="lightning_logs",
        ckpt_dir="checkpoints",
        num_workers=0  # Important for Windows compatibility
    )
    
    # 5. Use the trained model
    # The wrapped_model has all the methods of your original TorchModel
    test_data = np.random.randn(10, 128)
    predictions = wrapped_model.predict_on_batch(torch.tensor(test_data, dtype=torch.float32))
    
    # You can also compute uncertainty if your model supports it
    uncertainty = wrapped_model.predict_uncertainty(test_data)
    
    # 6. Save and load
    trainer.save_checkpoint("final_model.ckpt")
    
    # Later, load the model
    loaded_model = fsdp_wrapper.load_model(
        checkpoint_path="final_model.ckpt",
        torch_model_class=WeaveModel,
        model_args={'input_dim': 128, 'hidden_dim': 256}
    )
    
    return trainer, wrapped_model
if __name__ == "__main__":
    example_usage()
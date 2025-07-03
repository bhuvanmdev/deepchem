import pytest
import torch
import numpy as np
import lightning as L
import deepchem as dc
from copy import deepcopy
from pathlib import Path
import multiprocessing

# Import the custom modules you are testing
from deepchem.models.lightning.new_dc_lightning_dataset_module import DeepChemLightningDataModule
from deepchem.models.lightning.new_dc_lightning_module import DeepChemLightningModule

# Check if a GPU is available for testing, otherwise skip GPU tests
try:
    import torch.cuda
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
except ImportError:
    gpu_available = False

pytestmark = pytest.mark.skipif(not gpu_available, reason="Tests require a GPU.")

# Or use this approach for pytest
# @pytest.fixture(scope="session", autouse=True)
# def setup_multiprocessing():
#     if torch.cuda.is_available():
#         try:
#             multiprocessing.set_start_method('spawn', force=True)
#         except RuntimeError:
#             pass  # Already set


@pytest.fixture(scope="module")
def gcn_data():
    """
    Fixture to load the BACE dataset for a GCNModel.
    This runs only once per test module, saving time.
    """
    featurizer = dc.feat.MolGraphConvFeaturizer()
    tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer=featurizer)
    train_dataset, valid_dataset, _ = datasets
    
    # Using the validation set for faster testing, as in the reference file
    return {"dataset": valid_dataset, "n_tasks": len(tasks), "transformers": transformers,"train_dataset": train_dataset}

@pytest.fixture(scope="function")
def gcn_model(gcn_data):
    """
    Fixture to create a fresh GCNModel for each test function.
    This ensures tests are independent and don't share a trained state.
    """
    return dc.models.GCNModel(
        mode='classification',
        n_tasks=gcn_data["n_tasks"],
        batch_size=16,
        learning_rate=0.001
    )


# def test_gcn_fit_predict_workflow(gcn_model, gcn_data):
#     """
#     Tests if the fit and predict workflow works for a GCNModel.
#     This validates the custom DataModule's collate function and the LightningModule's
#     training and prediction steps with complex graph data.
#     """
#     dataset = gcn_data["dataset"]

#     # 1. Setup DataModule and LightningModule
#     # The `gcn_model` instance is passed to the DataModule to handle the complex collation
#     data_module = DeepChemLightningDataModule(dataset=dataset, batch_size=16, model=gcn_model)
#     lightning_model = DeepChemLightningModule(model=gcn_model)

#     # 2. Setup Trainer
#     # fast_dev_run executes a single batch for train/val/predict, which is ideal for testing
#     trainer = L.Trainer(
#         max_epochs=1,
#         accelerator="gpu",
#         devices=-1, # use all available GPUs
#         logger=False,
#         enable_checkpointing=False,
#         enable_progress_bar=False,
#         fast_dev_run=True,
#     )

#     # 3. Test fit
#     # This will fail if train_dataloader, collate_fn, or training_step has an issue
#     trainer.fit(model=lightning_model, datamodule=data_module)
    
#     # 4. Test predict
#     # This will fail if predict_dataloader or predict_step has an issue
#     prediction_batches = trainer.predict(model=lightning_model, datamodule=data_module)
    
#     # For classification, GCNModel outputs logits of shape (batch, tasks, classes)
#     # The trainer returns a list of outputs from each batch, so we concatenate them
#     predictions = np.concatenate([p for p in prediction_batches])

#     # 5. Verify prediction output
#     assert isinstance(prediction_batches, list)
#     assert len(prediction_batches) > 0

#     assert isinstance(predictions, np.ndarray)
#     # The final prediction shape should be (n_samples, n_tasks, n_classes=2), but since N_tasks=1 its shape is (n_samples, 2)
#     # Note: fast_dev_run uses a batch size of 16, so n_samples will be 16
#     assert predictions.shape == (16, 2)


# def test_gcn_checkpointing_and_reloading(gcn_model, gcn_data, tmp_path="temp"):
#     """
#     Tests that a GCNModel can be saved via a checkpoint and reloaded correctly.
#     It verifies that the model state is identical before saving and after loading.
#     """
#     dataset = gcn_data["dataset"]
    
#     # 1. Setup modules and trainer with a temporary directory for checkpoints
#     data_module = DeepChemLightningDataModule(dataset=dataset, batch_size=16, model=gcn_model)
#     lightning_model = DeepChemLightningModule(model=gcn_model)

#     trainer = L.Trainer(
#         max_epochs=10,
#         accelerator="gpu",
#         devices=-1, # use all available GPUs
#         default_root_dir=str(tmp_path), # Save checkpoints to a temp dir
#         enable_progress_bar=False,
#     )

#     # 2. Store model state *before* training for comparison
#     state_before_training = deepcopy(lightning_model.model.state_dict())

#     # 3. Train the model for one epoch, which will create a checkpoint
#     trainer.fit(model=lightning_model, datamodule=data_module)

#     # 4. Get model state *after* training
#     state_after_training = lightning_model.model.state_dict()

#     # --- Correctness Check 1: Before Saving ---
#     # Verify that training actually changed the model's weights
#     weight_changed = False
#     for key in state_before_training:
#         if not torch.allclose(state_before_training[key].detach().cpu(), state_after_training[key].detach().cpu()):
#             weight_changed = True
#             break
#     assert weight_changed, "Model weights did not change after one epoch of training."

#     # 5. Find the saved checkpoint file
#     checkpoint_dir = Path(tmp_path) / "lightning_logs" / "version_0" / "checkpoints"
#     checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
#     assert len(checkpoint_files) > 0, "Checkpoint file was not created."
#     ckpt_path = checkpoint_files[0]

#     # 6. Load the model from the checkpoint
#     # This is the standard Lightning way to reload a model
#     # We must pass the original `gcn_model` to properly re-initialize the DeepChem-specific parts
#     reloaded_model = DeepChemLightningModule.load_from_checkpoint(
#         ckpt_path, model=gcn_model
#     )
#     state_reloaded = reloaded_model.model.state_dict()

#     # --- Correctness Check 2: After Loading ---
#     # A. Verify that the reloaded state dict has the same keys
#     assert state_after_training.keys() == state_reloaded.keys()

#     # B. Verify that the reloaded weights are identical to the saved weights
#     for key in state_after_training:
#         torch.testing.assert_close(
#             state_after_training[key].detach().cpu(),
#             state_reloaded[key].detach().cpu(),
#             msg=f"Weight mismatch for key {key} after reloading.",
#         )

#     # --- Correctness Check 3: Functional Equivalence ---
#     # Predict with both models and compare results to ensure they are identical
#     original_preds_batches = trainer.predict(lightning_model, datamodule=data_module)
#     reloaded_preds_batches = trainer.predict(reloaded_model, datamodule=data_module)

#     original_preds = np.concatenate([p[0] for p in original_preds_batches])
#     reloaded_preds = np.concatenate([p[0] for p in reloaded_preds_batches])

#     np.testing.assert_allclose(
#         original_preds,
#         reloaded_preds,
#         err_msg="Predictions from original and reloaded models do not match.",
#     )


def test_gcn_overfit_with_lightning_trainer(gcn_model, gcn_data):
    """
    Tests if the GCN model can overfit to a small dataset using Lightning trainer.
    This validates that the Lightning training loop works correctly and the model
    can learn from the data by achieving good performance on the training set.
    Also tests the new multi-GPU prediction capability.
    """
    import lightning as L
    from deepchem.models.lightning.trainer2 import DeepChemLightningTrainer
    from deepchem.models.tests.test_graph_models import get_dataset
    from deepchem.feat import MolGraphConvFeaturizer
    import numpy as np
    from pathlib import Path
    L.seed_everything(42)
    np.random.seed(42)  # Ensure reproducibility for numpy operations
    torch.manual_seed(42)  # Ensure reproducibility for PyTorch operations
    # Use the same synthetic dataset approach as the reference test
    # Use num_tasks=1 to match the fixture gcn_model which has n_tasks=1
    tasks, dataset, transformers, metric = get_dataset('classification', featurizer=MolGraphConvFeaturizer(), num_tasks=1)

    # Create a new GCN model with the same configuration as the reference test
    gcn_model = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),  # This will be 1 now
        number_atom_features=30,  # Same as reference
        batch_size=10,  # Same as reference
        learning_rate=0.0003,  # Same as reference
        device='cpu',  # Use GPU for training
    )

    # Create Lightning trainer with parameters similar to reference test
    # Define a custom checkpoint directory
    checkpoint_dir = "my_custom_checkpoints"
    lightning_trainer = DeepChemLightningTrainer(
        model=gcn_model,
        batch_size=10,  # Same as reference
        max_epochs=100,  # Reduce for debugging
        accelerator="cuda",
        strategy="fsdp",
        devices=-1,  # Use multiple GPUs to test the multi-GPU prediction fix
        logger=False,
        enable_progress_bar=False,
        default_root_dir=checkpoint_dir,  # Save checkpoints to this directory
    )
    dataset = dc.data.DiskDataset.from_numpy(dataset.X, dataset.y, dataset.w, dataset.ids)
    
    # Debug: Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset y shape: {dataset.y.shape}")
    print(f"Dataset w shape: {dataset.w.shape}")

    # Train the model
    lightning_trainer.fit(dataset)

    # After training, create a new DeepChemLightningTrainer instance for prediction and load the best checkpoint
    # Find the latest checkpoint
    lightning_trainer.save_checkpoint("best_model.ckpt")

    # Create a new model instance and load weights
    gcn_model_pred = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        number_atom_features=30,
        batch_size=10,
        learning_rate=0.0003,
        device='cpu',
    )
    # Load weights from checkpoint
    lightning_trainer_pred = DeepChemLightningTrainer.load_checkpoint("best_model.ckpt", gcn_model_pred, 10,accelerator="cuda",
        strategy="fsdp",
        devices=-1,
        logger=False,
        enable_progress_bar=False,
        default_root_dir=checkpoint_dir,)

    # Test multi-GPU prediction directly
    print(f"Testing multi-GPU prediction...")
    # try:
    predictions_multi = lightning_trainer_pred.predict(dataset)
    print(f"Multi-GPU prediction successful!")
    print(f"Predictions type: {type(predictions_multi)}")
    if isinstance(predictions_multi, list) and len(predictions_multi) > 0:
        total_samples = sum(getattr(p, 'shape', [0])[0] for p in predictions_multi)
        print(f"Total prediction samples (multi-GPU): {total_samples}")
        print(f"Expected dataset size: {len(dataset)}")
    # except Exception as e:
    # print(f"Multi-GPU prediction failed: {e}")
    # predictions_multi = None

    # Now test evaluation (which uses prediction internally)
    try:
        scores_multi = lightning_trainer_pred.evaluate(dataset, [metric], transformers)
        print(f"Multi-GPU evaluation successful!")
        print(f"Multi-GPU ROC score: {scores_multi.get('mean-roc_auc_score', 'N/A')}")
    except Exception as e:
        print(f"Multi-GPU evaluation failed: {e}")
        scores_multi = None

    assert scores_multi["mean-roc_auc_score"] > 0.85, "Model did not learn anything during training."
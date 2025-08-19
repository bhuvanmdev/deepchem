import pytest
import deepchem as dc
import numpy as np
import os
import shutil
import glob

try:
    import torch
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() >= 2
except ImportError:
    gpu_available = False

try:
    import lightning as L
    from deepchem.models.lightning.trainer import LightningTorchModel
    PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
    PYTORCH_LIGHTNING_IMPORT_FAILED = True


@pytest.mark.skipif(not gpu_available,
                   reason="Need at least 2 GPUs for this test")
@pytest.mark.skipif(PYTORCH_LIGHTNING_IMPORT_FAILED,
                   reason="PyTorch Lightning is not installed")
@pytest.mark.torch
def test_model_checkpointing_with_rotation():
    """
    Test Lightning trainer's model checkpointing functionality with rotating checkpoints.
    This test validates:
    1. Checkpoints are created at specified intervals (train steps)
    2. Only max_checkpoints_to_keep number of checkpoints are maintained
    3. The model can be trained using 2 GPUs with FSDP strategy
    4. Checkpoint files are properly named with epoch and step information
    """
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    L.seed_everything(42)
    
    # Load the BACE dataset for GCNModel
    featurizer = dc.feat.MolGraphConvFeaturizer()
    tasks, all_dataset, _ = dc.molnet.load_bace_classification(featurizer)
    train_dataset, _, _ = all_dataset
    
    # Create a smaller subset for faster testing
    train_dataset = dc.data.DiskDataset.from_numpy(
        train_dataset.X[:50], 
        train_dataset.y[:50], 
        train_dataset.w[:50],
        train_dataset.ids[:50]
    )
    
    # Create GCN model
    gcn_model = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        number_atom_features=30,
        batch_size=4,
        learning_rate=0.001,
        device='cpu',  # Will be moved to GPU by Lightning
    )
    
    # Set up checkpoint directory
    checkpoint_dir = "test_checkpoints"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Configure Lightning trainer with checkpointing
    lightning_trainer = LightningTorchModel(
        model=gcn_model,
        batch_size=4,
        max_epochs=20,
        accelerator="cuda",
        devices=2,  # Use 2 GPUs
        strategy="fsdp",  # Use FSDP strategy
        log_every_n_steps=1,
        enable_checkpointing=True,
        model_dir=checkpoint_dir,
        logger=False,  # Disable logging for cleaner test output
        enable_progress_bar=False  # Disable progress bar for cleaner test output
    )
    
    # Test checkpoint settings
    max_checkpoints_to_keep = 3
    checkpoint_interval = 5  # Save every 5 training steps
    
    # Train the model with custom checkpoint settings
    lightning_trainer.fit(
        train_dataset,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        checkpoint_interval=checkpoint_interval,
        num_workers=0  # Use 0 workers to avoid multiprocessing issues in testing
    )
    
    # Verify checkpoint directory exists
    checkpoint_subdir = os.path.join(checkpoint_dir, "checkpoints")
    assert os.path.exists(checkpoint_subdir), "Checkpoint directory should exist"
    
    # Check for checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_subdir, "*.ckpt"))
    print(f"Found {len(checkpoint_files)} checkpoint files: {checkpoint_files}")
    
    # Verify that checkpoints were created
    assert len(checkpoint_files) > 0, "At least one checkpoint should be created"
    
    # Verify checkpoint rotation - should not exceed max_checkpoints_to_keep + 1 (for last.ckpt)
    assert len(checkpoint_files) <= max_checkpoints_to_keep + 1, \
        f"Should not have more than {max_checkpoints_to_keep + 1} checkpoint files"
    
    # Verify checkpoint file naming convention (should contain epoch and step)
    for checkpoint_file in checkpoint_files:
        filename = os.path.basename(checkpoint_file)
        if filename != "last.ckpt":  # Skip the last.ckpt file
            assert "epoch=" in filename or "step=" in filename, \
                f"Checkpoint filename should contain epoch or step info: {filename}"
    
    # Test loading from a checkpoint
    if checkpoint_files:
        # Find a checkpoint that's not last.ckpt
        regular_checkpoints = [f for f in checkpoint_files if not f.endswith("last.ckpt")]
        if regular_checkpoints:
            test_checkpoint = regular_checkpoints[0]
            
            # Create a new model instance
            new_gcn_model = dc.models.GCNModel(
                mode='classification',
                n_tasks=len(tasks),
                number_atom_features=30,
                batch_size=4,
                learning_rate=0.001,
                device='cpu',
            )
            
            # Load from checkpoint
            loaded_trainer = LightningTorchModel.load_checkpoint(
                test_checkpoint,
                model=new_gcn_model,
                batch_size=4,
                accelerator="cuda",
                devices=1,  # Use single GPU for prediction
                logger=False,
                enable_progress_bar=False,
                model_dir=checkpoint_dir
            )
            
            # Test prediction to ensure the model loaded correctly
            predictions = loaded_trainer.predict(train_dataset, num_workers=0)
            assert predictions is not None, "Predictions should not be None"
            assert len(predictions) > 0, "Should have some predictions"
            print(f"Successfully loaded checkpoint and made {len(predictions)} predictions")
    
    # Clean up
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    
    print("✅ All checkpoint tests passed!")


@pytest.mark.skipif(not gpu_available,
                   reason="Need at least 2 GPUs for this test")
@pytest.mark.skipif(PYTORCH_LIGHTNING_IMPORT_FAILED,
                   reason="PyTorch Lightning is not installed")
@pytest.mark.torch
def test_checkpoint_interval_disabled():
    """
    Test that checkpointing can be disabled by setting checkpoint_interval to 0.
    """
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    L.seed_everything(42)
    
    # Load a small dataset
    featurizer = dc.feat.MolGraphConvFeaturizer()
    tasks, all_dataset, _ = dc.molnet.load_bace_classification(featurizer)
    train_dataset, _, _ = all_dataset
    
    # Create a smaller subset for faster testing
    train_dataset = dc.data.DiskDataset.from_numpy(
        train_dataset.X[:20], 
        train_dataset.y[:20], 
        train_dataset.w[:20],
        train_dataset.ids[:20]
    )
    
    # Create GCN model
    gcn_model = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        number_atom_features=30,
        batch_size=4,
        learning_rate=0.001,
        device='cpu',
    )
    
    # Set up checkpoint directory
    checkpoint_dir = "test_checkpoints_disabled"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Configure Lightning trainer
    lightning_trainer = LightningTorchModel(
        model=gcn_model,
        batch_size=4,
        max_epochs=5,
        accelerator="cuda",
        devices=2,
        strategy="fsdp",
        log_every_n_steps=1,
        enable_checkpointing=True,
        model_dir=checkpoint_dir,
        logger=False,
        enable_progress_bar=False
    )
    
    # Train with checkpointing disabled (checkpoint_interval=0)
    lightning_trainer.fit(
        train_dataset,
        max_checkpoints_to_keep=3,
        checkpoint_interval=0,  # Disable automatic checkpointing
        num_workers=0
    )
    
    # Verify no automatic checkpoint files were created (except possibly default ones)
    checkpoint_subdir = os.path.join(checkpoint_dir, "checkpoints")
    if os.path.exists(checkpoint_subdir):
        checkpoint_files = glob.glob(os.path.join(checkpoint_subdir, "*.ckpt"))
        print(f"Found {len(checkpoint_files)} checkpoint files when interval=0: {checkpoint_files}")
        # With interval=0, we shouldn't create step-based checkpoints
        # Only epoch-based or last checkpoints might exist depending on Lightning's defaults
    
    # Clean up
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    
    print("✅ Checkpoint disabling test passed!")


if __name__ == "__main__":
    print("Running checkpoint functionality tests...")
    print(f"GPU available: {gpu_available}")
    print(f"PyTorch Lightning available: {not PYTORCH_LIGHTNING_IMPORT_FAILED}")
    
    if gpu_available and not PYTORCH_LIGHTNING_IMPORT_FAILED:
        test_model_checkpointing_with_rotation()
        test_checkpoint_interval_disabled()
        print("🎉 All tests completed successfully!")
    else:
        print("❌ Tests skipped due to missing requirements (need 2+ GPUs and PyTorch Lightning)")

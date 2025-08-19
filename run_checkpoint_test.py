#!/usr/bin/env python3

"""
Test script to verify model checkpointing functionality with Lightning trainer.
This script can be run directly to test the checkpointing features.
"""

import os
import sys
import shutil
import glob

# Add deepchem to path
sys.path.insert(0, r'e:\dc\deepchem')

try:
    import deepchem as dc
    print("✓ DeepChem imported successfully")
except ImportError as e:
    print(f"✗ Failed to import DeepChem: {e}")
    sys.exit(1)

try:
    import torch
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    print(f"✓ PyTorch imported. GPU available: {gpu_available}, Device count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")
    sys.exit(1)

try:
    import lightning as L
    from deepchem.models.lightning.trainer import LightningTorchModel
    print("✓ Lightning and LightningTorchModel imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Lightning: {e}")
    sys.exit(1)


def test_checkpointing():
    """Test the checkpointing functionality."""
    print("\n🧪 Testing model checkpointing with rotation...")
    
    # Set seeds for reproducibility
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)
    L.seed_everything(42)
    
    # Load the BACE dataset for GCNModel
    try:
        featurizer = dc.feat.MolGraphConvFeaturizer()
        tasks, all_dataset, _ = dc.molnet.load_bace_classification(featurizer)
        train_dataset, _, _ = all_dataset
        print(f"✓ Loaded BACE dataset with {len(train_dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load BACE dataset: {e}")
        return False
    
    # Create a smaller subset for faster testing
    train_dataset = dc.data.DiskDataset.from_numpy(
        train_dataset.X[:30], 
        train_dataset.y[:30], 
        train_dataset.w[:30],
        train_dataset.ids[:30]
    )
    print(f"✓ Created subset with {len(train_dataset)} samples")
    
    # Create GCN model
    try:
        gcn_model = dc.models.GCNModel(
            mode='classification',
            n_tasks=len(tasks),
            number_atom_features=30,
            batch_size=4,
            learning_rate=0.001,
            device='cpu',
        )
        print("✓ Created GCN model")
    except Exception as e:
        print(f"✗ Failed to create GCN model: {e}")
        return False
    
    # Set up checkpoint directory
    checkpoint_dir = "test_checkpoints_demo"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"✓ Created checkpoint directory: {checkpoint_dir}")
    
    # Configure Lightning trainer with appropriate settings based on GPU availability
    trainer_kwargs = {
        'model': gcn_model,
        'batch_size': 4,
        'max_epochs': 10,
        'log_every_n_steps': 1,
        'enable_checkpointing': True,
        'model_dir': checkpoint_dir,
        'logger': False,
        'enable_progress_bar': True,  # Keep progress bar for demo
        'fast_dev_run': False
    }
    
    if gpu_available:
        trainer_kwargs.update({
            'accelerator': "cuda",
            'devices': 2,
            'strategy': "fsdp"
        })
        print("✓ Configured for 2 GPUs with FSDP strategy")
    else:
        trainer_kwargs.update({
            'accelerator': "cpu",
            'devices': 1
        })
        print("✓ Configured for CPU training (no GPUs detected)")
    
    try:
        lightning_trainer = LightningTorchModel(**trainer_kwargs)
        print("✓ Created Lightning trainer")
    except Exception as e:
        print(f"✗ Failed to create Lightning trainer: {e}")
        return False
    
    # Test checkpoint settings
    max_checkpoints_to_keep = 3
    checkpoint_interval = 3  # Save every 3 training steps
    print(f"✓ Checkpoint settings: max_keep={max_checkpoints_to_keep}, interval={checkpoint_interval}")
    
    # Train the model with custom checkpoint settings
    try:
        print("\n🚀 Starting training...")
        lightning_trainer.fit(
            train_dataset,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            checkpoint_interval=checkpoint_interval,
            num_workers=0
        )
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # Verify checkpoint directory exists
    checkpoint_subdir = os.path.join(checkpoint_dir, "checkpoints")
    if not os.path.exists(checkpoint_subdir):
        print(f"✗ Checkpoint directory does not exist: {checkpoint_subdir}")
        return False
    print(f"✓ Checkpoint directory exists: {checkpoint_subdir}")
    
    # Check for checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_subdir, "*.ckpt"))
    print(f"✓ Found {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"  - {os.path.basename(f)}")
    
    # Verify that checkpoints were created
    if len(checkpoint_files) == 0:
        print("✗ No checkpoint files were created")
        return False
    
    # Verify checkpoint rotation
    if len(checkpoint_files) > max_checkpoints_to_keep + 1:
        print(f"✗ Too many checkpoint files: {len(checkpoint_files)} > {max_checkpoints_to_keep + 1}")
        return False
    print(f"✓ Checkpoint rotation working correctly")
    
    # Verify checkpoint file naming convention
    for checkpoint_file in checkpoint_files:
        filename = os.path.basename(checkpoint_file)
        if filename != "last.ckpt":
            if not ("epoch=" in filename or "step=" in filename):
                print(f"✗ Invalid checkpoint filename: {filename}")
                return False
    print("✓ Checkpoint filenames follow correct convention")
    
    # Test loading from a checkpoint
    regular_checkpoints = [f for f in checkpoint_files if not f.endswith("last.ckpt")]
    if regular_checkpoints:
        test_checkpoint = regular_checkpoints[0]
        print(f"\n🔄 Testing checkpoint loading from: {os.path.basename(test_checkpoint)}")
        
        try:
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
            load_kwargs = {
                'model': new_gcn_model,
                'batch_size': 4,
                'logger': False,
                'enable_progress_bar': False,
                'model_dir': checkpoint_dir
            }
            
            if gpu_available:
                load_kwargs.update({
                    'accelerator': "cuda",
                    'devices': 1
                })
            else:
                load_kwargs.update({
                    'accelerator': "cpu",
                    'devices': 1
                })
            
            loaded_trainer = LightningTorchModel.load_checkpoint(
                test_checkpoint,
                **load_kwargs
            )
            print("✓ Successfully loaded checkpoint")
            
            # Test prediction
            predictions = loaded_trainer.predict(train_dataset, num_workers=0)
            if predictions is None or len(predictions) == 0:
                print("✗ Predictions failed or empty")
                return False
            print(f"✓ Successfully made {len(predictions)} predictions from loaded checkpoint")
            
        except Exception as e:
            print(f"✗ Checkpoint loading failed: {e}")
            return False
    
    # Clean up
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"✓ Cleaned up checkpoint directory")
    
    return True


# if __name__ == "__main__":
#     print("=== Model Checkpointing Test ===")
#     print("This test validates the Lightning trainer's checkpointing functionality.")
#     print("It will train a GCN model and verify that checkpoints are created and rotated correctly.")
    
#     if gpu_available:
#         print(f"\n🎮 GPU mode: Using {torch.cuda.device_count()} GPUs with FSDP strategy")
#     else:
#         print("\n💻 CPU mode: No suitable GPUs detected, falling back to CPU training")
    
#     success = test_checkpointing()
    
#     if success:
#         print("\n🎉 All checkpoint tests passed successfully!")
#         print("\nKey features verified:")
#         print("✓ Checkpoints saved at specified intervals (training steps)")
#         print("✓ Checkpoint rotation maintains max_checkpoints_to_keep limit")
#         print("✓ Checkpoint files follow proper naming convention")
#         print("✓ Model can be loaded from checkpoint and used for prediction")
#         if gpu_available:
#             print("✓ Multi-GPU training with FSDP strategy works correctly")
#         sys.exit(0)
#     else:
#         print("\n❌ Checkpoint tests failed!")
#         sys.exit(1)

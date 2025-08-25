import pytest
import deepchem as dc
import numpy as np
import os
import shutil
try:
    import torch
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
except ImportError:
    gpu_available = False

try:
    import lightning as L
    from deepchem.models.lightning.trainer import LightningTorchModel
    PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
    PYTORCH_LIGHTNING_IMPORT_FAILED = True

pytestmark = [
    pytest.mark.skipif(not gpu_available,
                       reason="No GPU available for testing"),
    pytest.mark.skipif(PYTORCH_LIGHTNING_IMPORT_FAILED,
                       reason="PyTorch Lightning is not installed")
]


@pytest.mark.torch
def test_default_restore():
    """Test restore method using automatic checkpoints created during fit()."""
    L.seed_everything(42)
    tasks, datasets, _ = dc.molnet.load_clintox()
    _, valid_dataset, _ = datasets

    # Create first model and trainer with automatic checkpointing
    model1 = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                           n_features=1024,
                                           layer_sizes=[1000],
                                           dropouts=0.2,
                                           learning_rate=0.0001,
                                           device="cpu",
                                           batch_size=16)

    # Use a specific model_dir for this test to avoid conflicts
    trainer1 = LightningTorchModel(model=model1,
                                   batch_size=16,
                                   model_dir="test_default_restore_dir",
                                   accelerator="cuda",
                                   devices=-1,
                                   log_every_n_steps=1)

    # Train first model - this will automatically create checkpoints
    trainer1.fit(valid_dataset, nb_epoch=3, checkpoint_interval=20)

    y1 = trainer1.predict(valid_dataset)
    
    # Create second model and trainer with same model_dir
    model2 = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                           n_features=1024,
                                           layer_sizes=[1000],
                                           dropouts=0.2,
                                           learning_rate=0.0001,
                                           device="cpu",
                                           batch_size=16)

    trainer2 = LightningTorchModel(model=model2,
                                   batch_size=16,
                                   model_dir="test_default_restore_dir",
                                   accelerator="cuda",
                                   devices=-1,
                                   log_every_n_steps=1)

    # Restore from automatic checkpoint (should find last.ckpt in checkpoints dir)
    trainer2.restore()

    # Now they should produce similar results
    y2 = trainer2.predict(valid_dataset)
    assert np.allclose(y1, y2, atol=1e-3)

    # Clean up
    if os.path.exists("test_default_restore_dir"):
        shutil.rmtree("test_default_restore_dir")

@pytest.mark.torch
def test_multitask_classifier_restore_correctness():
    L.seed_everything(42)
    tasks, datasets, _ = dc.molnet.load_clintox()
    _, valid_dataset, _ = datasets

    model = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                          n_features=1024,
                                          layer_sizes=[1000],
                                          dropouts=0.2,
                                          learning_rate=0.0001,
                                          device="cpu",
                                          batch_size=16)

    trainer = LightningTorchModel(model=model,
                                  batch_size=16,
                                  model_dir="test_multitask_restore_dir",
                                  accelerator="cuda",
                                  devices=-1,
                                  log_every_n_steps=1,
                                  strategy="fsdp")

    trainer.fit(valid_dataset, nb_epoch=3)
    # get a some 10 weights for assertion
    weights = trainer.model.model.layers[0].weight[:10].detach().cpu().numpy()


    # Create new model and trainer instance
    restore_model = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                                  n_features=1024,
                                                  layer_sizes=[1000],
                                                  dropouts=0.2,
                                                  learning_rate=0.0001,
                                                  device="cpu",
                                                  batch_size=16)

    trainer2 = LightningTorchModel(model=restore_model,
                                   batch_size=16,
                                   model_dir="test_multitask_restore_dir",
                                   accelerator="cuda",
                                   devices=-1,
                                   log_every_n_steps=1,
                                   strategy="fsdp")

    # Restore from checkpoint
    trainer2.restore()

    # get a some 10 weights for assertion
    reloaded_weights = trainer2.model.model.layers[0].weight[:10].detach().cpu().numpy()

    _ = trainer2.predict(valid_dataset)

    # make it equal with a tolerance of 1e-5
    assert torch.allclose(torch.tensor(weights),
                          torch.tensor(reloaded_weights),
                          atol=1e-5)

    # Clean up
    if os.path.exists("test_multitask_restore_dir"):
        shutil.rmtree("test_multitask_restore_dir")


@pytest.mark.torch
def test_gcn_model_restore_correctness():
    L.seed_everything(42)
    featurizer = dc.feat.MolGraphConvFeaturizer()
    tasks, all_dataset, _ = dc.molnet.load_bace_classification(featurizer)
    _, valid_dataset, _ = all_dataset

    model = dc.models.GCNModel(mode='classification',
                               n_tasks=len(tasks),
                               batch_size=16,
                               learning_rate=0.001,
                               device="cpu")

    trainer = LightningTorchModel(model=model,
                                  batch_size=16,
                                  model_dir="test_gcn_restore_dir",
                                  accelerator="cuda",
                                  devices=-1,
                                  log_every_n_steps=1,
                                  strategy="fsdp")

    trainer.fit(valid_dataset, nb_epoch=3)

    # get a some 10 weights for assertion
    weights = trainer.model.model.model.gnn.gnn_layers[
        0].res_connection.weight[:10].detach().cpu().numpy()


    # Create new model and trainer instance
    restore_model = dc.models.GCNModel(mode='classification',
                                      n_tasks=len(tasks),
                                      batch_size=16,
                                      learning_rate=0.001,
                                      device="cpu")

    trainer2 = LightningTorchModel(model=restore_model,
                                   batch_size=16,
                                   model_dir="test_gcn_restore_dir",
                                   accelerator="cuda",
                                   devices=-1,
                                   log_every_n_steps=1,
                                   strategy="fsdp")

    # Restore from checkpoint - look for checkpoint1.ckpt in model_dir
    trainer2.restore()

    # get a some 10 weights for assertion
    reloaded_weights = trainer2.model.model.model.gnn.gnn_layers[
        0].res_connection.weight[:10].detach().cpu().numpy()

    _ = trainer2.predict(valid_dataset)

    # make it equal with a tolerance of 1e-5
    assert torch.allclose(torch.tensor(weights),
                          torch.tensor(reloaded_weights),
                          atol=1e-5)

    # Clean up
    if os.path.exists("test_gcn_restore_dir"):
        shutil.rmtree("test_gcn_restore_dir")


@pytest.mark.torch
def test_gcn_overfit_with_lightning_trainer():
    """
    Tests if the GCN model can overfit to a small dataset using Lightning trainer.
    This validates that the Lightning training loop works correctly and the model
    can learn from the data.
    """

    np.random.seed(42)  # Ensure reproducibility for numpy operations
    torch.manual_seed(42)  # Ensure reproducibility for PyTorch operations
    L.seed_everything(42)

    # Load the BACE dataset for GCNModel
    from deepchem.models.tests.test_graph_models import get_dataset
    from deepchem.feat import MolGraphConvFeaturizer
    tasks, dataset, transformers, metric = get_dataset(
        'classification', featurizer=MolGraphConvFeaturizer())
    dataset = dc.data.DiskDataset.from_numpy(dataset.X, dataset.y, dataset.w,
                                             dataset.ids)
    gcn_model = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        number_atom_features=30,
        batch_size=5,
        learning_rate=0.0003,
        device='cpu',
    )

    lightning_trainer = LightningTorchModel(
        model=gcn_model,
        batch_size=5,
        accelerator="cuda",
        strategy="fsdp",
        devices=-1,
        enable_checkpointing=True)

    # Train the model
    lightning_trainer.fit(dataset,
                          max_checkpoints_to_keep=3,
                          checkpoint_interval=20,
                          nb_epoch=70)

    # evaluate the checkpoints availablity 3 + 1 represents the 3 checkpoints plus the final model
    checkpoints = os.listdir(os.path.join(lightning_trainer.model_dir, "checkpoints"))
    assert len(
        checkpoints) == 3 + 1, "No checkpoints were created during training."

    # After training, create a new LightningTorchModel instance for prediction and load the best checkpoint

    # Create a new model instance and load weights
    gcn_model_pred = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        number_atom_features=30,
        batch_size=10,
        learning_rate=0.0003,
        device='cpu',
    )

    # Load weights from checkpoint using the same model_dir
    lightning_trainer_pred = LightningTorchModel(
        model=gcn_model_pred,
        batch_size=10,
        model_dir=lightning_trainer.model_dir,
        accelerator="cuda",
        logger=False,
        enable_progress_bar=False)
    
    lightning_trainer_pred.restore()

    scores_multi = lightning_trainer_pred.evaluate(dataset, [metric],
                                                   transformers)
    
    assert scores_multi[
        "mean-roc_auc_score"] > 0.85, "Model did not learn anything during training."

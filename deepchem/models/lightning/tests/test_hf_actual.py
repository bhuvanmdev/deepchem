import os
# Set tokenizers parallelism to false BEFORE any imports to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytest
import numpy as np
import deepchem as dc
from deepchem.data import DiskDataset
from copy import deepcopy
from pathlib import Path
import os
try:
    import torch
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
except ImportError:
    gpu_available = False

try:
    import lightning as L
    from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule
    from deepchem.models.lightning.dc_lightning_module import DCLightningModule
    from deepchem.models.lightning.trainer import DeepChemLightningTrainer
    PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
    PYTORCH_LIGHTNING_IMPORT_FAILED = True

try:
    from deepchem.models.torch_models.chemberta import Chemberta
    CHEMBERTA_IMPORT_FAILED = False
except ImportError:
    CHEMBERTA_IMPORT_FAILED = True

# pytestmark = [
#     pytest.mark.skipif(not gpu_available,
#                        reason="No GPU available for testing"),
#     pytest.mark.skipif(torch.cuda.device_count() < 2,
#                        reason="FSDP testing requires at least 2 GPUs"),
#     pytest.mark.skipif(PYTORCH_LIGHTNING_IMPORT_FAILED,
#                        reason="PyTorch Lightning is not installed"),
#     pytest.mark.skipif(CHEMBERTA_IMPORT_FAILED,
#                        reason="ChemBERTa is not installed")
# ]


@pytest.fixture(scope="function")
def smiles_data(tmp_path_factory):
    """
    Fixture to create a small SMILES dataset for ChemBERTa testing.
    """
    # Small set of SMILES strings for testing - increased for FSDP
    smiles = [
        'CCO',  
        'CCC',  
        'CC(C)O',
        'C1=CC=CC=C1',
        'CCN(CC)CC',
        'CC(=O)O',
        'C1=CC=C(C=C1)O',
        'CCO[Si](OCC)(OCC)OCC',
        'CC(C)(C)O',
        'CC(C)C',
        'CCC(C)C',
        'CCCC',
        'CCCCC',
        'CCCCCC',
        'CC(C)CC',
        'CCC(C)(C)C'
    ]
    
    # molecular weight predictions
    labels = [46.07, 44.10, 60.10, 78.11, 101.19, 60.05, 94.11, 208.33,
              74.12, 58.12, 72.15, 58.12, 72.15, 86.18, 72.15, 86.18]
    

    data_dir = tmp_path_factory.mktemp("dataset")
    
    dataset = DiskDataset.from_numpy(
        X=np.array(smiles), 
        y=np.array(labels).reshape(-1, 1),
        w=np.ones(len(smiles)),
        ids=np.arange(len(smiles)),
        data_dir=str(data_dir)
    )
    
    return {
        "dataset": dataset,
        "n_tasks": 1,
        "smiles": smiles,
        "labels": labels
    }


@pytest.fixture(scope="function") 
def chemberta_tokenizer():
    """
    Fixture to load ChemBERTa tokenizer.
    """
    # Use a smaller, faster ChemBERTa model for testing
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    tokenizer = ''#AutoTokenizer.from_pretrained(model_name)
    return tokenizer


@pytest.mark.torch
def test_chemberta_masked_lm_workflow(smiles_data, chemberta_tokenizer):
    """
    Tests the masked language modeling workflow with ChemBERTa using FSDP.
    This validates the Lightning wrappers work with HuggingFace transformer models
    for self-supervised pretraining tasks with FSDP distributed training.
    """
    dataset = smiles_data["dataset"]
    # model_name = "seyonec/ChemBERTa-zinc-base-v1"
    
    # Load ChemBERTa model for masked language modeling
  
    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"

    dc_hf_model =  Chemberta(task='mlm', tokenizer_path=tokenizer_path, device='cpu', batch_size=2, learning_rate=0.0001)
    
    # Setup DeepChemLightningTrainer for MLM pretraining with FSDP
    trainer = DeepChemLightningTrainer(
        model=dc_hf_model,
        batch_size=2,  # Increased batch size for FSDP
        max_epochs=1,
        accelerator="gpu",
        devices=-1,  # Use all available GPUs
        strategy="fsdp",  # Enable FSDP strategy
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        fast_dev_run=True,
        # precision="16-mixed",  # Mixed precision for efficiency
    )
    
    # Test MLM training
    trainer.fit(train_dataset=dataset, num_workers=4)
    
    trainer.save_checkpoint("chemberta_mlm_checkpoint.ckpt")

    # Create a new model instance for loading
    new_dc_hf_model = Chemberta(task='mlm', tokenizer_path=tokenizer_path, device='cpu', batch_size=2, learning_rate=0.0001)
    
    # Load the checkpoint into the new model instance
    reloaded_trainer = DeepChemLightningTrainer.load_checkpoint(
        "chemberta_mlm_checkpoint.ckpt", model=new_dc_hf_model,
        batch_size=2,
        accelerator="gpu",
        devices=1,
    )

    # Test MLM prediction (masked token prediction) using the reloaded trainer
    prediction_batches = reloaded_trainer.predict(dataset=dataset, num_workers=0)
    
    # Verify prediction output
    assert isinstance(prediction_batches, list)
    assert len(prediction_batches) > 0
    
    # For MLM, predictions should be token logits
    if prediction_batches and prediction_batches[0] is not None:
        predictions = prediction_batches[0]
        assert isinstance(predictions, (torch.Tensor, np.ndarray))


@pytest.mark.torch
def test_chemberta_regression_workflow(smiles_data, chemberta_tokenizer):
    """
    Tests the regression workflow with ChemBERTa for molecular property prediction using FSDP.
    This validates the Lightning wrappers work with HuggingFace models for downstream tasks
    with FSDP distributed training.
    """
    dataset = smiles_data["dataset"]
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    
    # Load ChemBERTa model for sequence classification (configured for regression)
    # hf_model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name,
    #     num_labels=1,  # regression with single output
    #     problem_type='regression'
    # )
    
    # # Wrap in DeepChem HuggingFaceModel
    # dc_hf_model = HuggingFaceModel(
    #     model=hf_model,
    #     tokenizer=chemberta_tokenizer,
    #     task='regression',
    #     batch_size=8,  # Increased batch size for FSDP
    #     learning_rate=0.0001,
    #     device='cpu'  # Device will be managed by Lightning/FSDP
    # )
    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"

    dc_hf_model =  Chemberta(task='regression', tokenizer_path=tokenizer_path, device='cpu', batch_size=2, learning_rate=0.0001)
    # Setup DeepChemLightningTrainer for regression training with FSDP
    trainer = DeepChemLightningTrainer(
        model=dc_hf_model,
        batch_size=8,  # Increased batch size for FSDP
        max_epochs=1,
        accelerator="gpu",
        devices=-1,  # Use all available GPUs
        strategy="fsdp",  # Enable FSDP strategy
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        fast_dev_run=True,
        precision="16-mixed",  # Mixed precision for efficiency
    )
    
    # Test regression training
    trainer.fit(train_dataset=dataset, num_workers=4)
    trainer.save_checkpoint("chemberta_reg_checkpoint.ckpt")

    # Create a new model instance for loading
    new_dc_hf_model = Chemberta(task='regression', tokenizer_path=tokenizer_path, device='cpu', batch_size=2, learning_rate=0.0001)
    
    # Load the checkpoint into the new model instance
    reloaded_trainer = DeepChemLightningTrainer.load_checkpoint(
        "chemberta_reg_checkpoint.ckpt", model=new_dc_hf_model,
        batch_size=2,
        accelerator="gpu",
        devices=1,
    )

    # Test regression prediction using the reloaded trainer
    prediction_batches = reloaded_trainer.predict(dataset=dataset, num_workers=0)
    
    # Verify prediction output
    assert isinstance(prediction_batches, list)
    assert len(prediction_batches) > 0
    
    # Handle potential None values in prediction batches for FSDP
    valid_predictions = [p for p in prediction_batches if p is not None]
    if valid_predictions:
        predictions = np.concatenate(valid_predictions)
        assert isinstance(predictions, np.ndarray)
        
        # For regression, predictions should match the number of samples and tasks
        assert predictions.shape[1] == 1  # single regression task


@pytest.mark.torch 
def test_chemberta_classification_workflow(smiles_data, chemberta_tokenizer, tmp_path):
    """
    Tests the classification workflow with ChemBERTa for molecular classification using FSDP.
    """
    dataset = smiles_data["dataset"]
    
    # Convert regression labels to binary classification
    y_binary = (dataset.y > np.median(dataset.y)).astype(int)
    
    # Create DiskDataset for classification instead of NumpyDataset
    classification_dataset = DiskDataset.from_numpy(
        X=dataset.X, 
        y=y_binary,
        w=dataset.w,
        ids=dataset.ids,
        data_dir=str(tmp_path / "classification_data")
    )
    
    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"
    
    # Load ChemBERTa model for binary classification using Chemberta
    dc_hf_model = Chemberta(task='classification', tokenizer_path=tokenizer_path, device='cpu', batch_size=2, learning_rate=0.0001)
    
    # Setup DeepChemLightningTrainer for classification training with FSDP
    trainer = DeepChemLightningTrainer(
        model=dc_hf_model,
        batch_size=2,  # Match model batch size
        max_epochs=1,
        accelerator="gpu", 
        devices=-1,  # Use all available GPUs
        strategy="fsdp",  # Enable FSDP strategy
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        fast_dev_run=True,
        # precision="16-mixed",  # Mixed precision for efficiency
    )
    
    # Test classification training
    trainer.fit(train_dataset=classification_dataset, num_workers=4)
    
    trainer.save_checkpoint("chemberta_classification_checkpoint.ckpt")

    # Create a new model instance for loading
    new_dc_hf_model = Chemberta(task='classification', tokenizer_path=tokenizer_path, device='cpu', batch_size=2, learning_rate=0.0001)
    
    # Load the checkpoint into the new model instance
    reloaded_trainer = DeepChemLightningTrainer.load_checkpoint(
        "chemberta_classification_checkpoint.ckpt", model=new_dc_hf_model,
        batch_size=2,
        accelerator="gpu",
        devices=1,
    )

    # Test classification prediction using the reloaded trainer
    prediction_batches = reloaded_trainer.predict(dataset=classification_dataset, num_workers=0)
    
    # Verify prediction output
    assert isinstance(prediction_batches, list)
    assert len(prediction_batches) > 0
    
    # Handle potential None values in prediction batches for FSDP
    valid_predictions = [p for p in prediction_batches if p is not None]
    if valid_predictions:
        predictions = np.concatenate(valid_predictions)
        assert isinstance(predictions, np.ndarray)
        
        # For binary classification, predictions should be probabilities for 2 classes
        assert predictions.shape[1] == 2  # binary classification probabilities


@pytest.mark.torch
def test_chemberta_checkpointing_and_loading(smiles_data, chemberta_tokenizer, tmp_path="temp"):
    """
    Tests that a ChemBERTa model can be saved via a checkpoint and reloaded correctly with FSDP.
    It verifies that the model state is identical before saving and after loading in FSDP scenarios.
    """
    dataset = smiles_data["dataset"]
    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"
    
    # Load ChemBERTa model for regression using Chemberta
    dc_hf_model = Chemberta(task='regression', tokenizer_path=tokenizer_path, device='cpu', batch_size=2, learning_rate=0.0001)
    
    # Setup trainer with temporary directory for checkpoints
    trainer = DeepChemLightningTrainer(
        model=dc_hf_model,
        batch_size=2,  # Match model batch size
        max_epochs=1,
        accelerator="gpu",
        devices=-1,
        default_root_dir=str(tmp_path),
        enable_progress_bar=False,
        logger=False,
        # precision="16-mixed",  # Mixed precision for efficiency
    )
    
    # Store model state *before* training for comparison
    state_before_training = deepcopy(trainer.lightning_model.pt_model.state_dict())
    
    # Train the model for one epoch, which will create a checkpoint
    trainer.fit(train_dataset=dataset, num_workers=4)
    
    # Get model state *after* training
    state_after_training = trainer.lightning_model.pt_model.state_dict()
    
    # --- Correctness Check 1: Before Saving ---
    # Verify that training actually changed the model's weights
    weight_changed = False
    for key in state_before_training:
        if not torch.allclose(state_before_training[key].detach().cpu(),
                              state_after_training[key].detach().cpu(), rtol=1e-4, atol=1e-6):
            weight_changed = True
            break
    assert weight_changed, "Model weights did not change after one epoch of training."
    
    # Save checkpoint using DeepChemLightningTrainer
    checkpoint_path = str(Path(tmp_path) / "model_checkpoint.ckpt")
    trainer.save_checkpoint(checkpoint_path)
    
    # Create a new model instance for loading
    dc_hf_model_new = Chemberta(task='regression', tokenizer_path=tokenizer_path, device='cpu', batch_size=2, learning_rate=0.0001)
    
    # Load the model from the checkpoint using DeepChemLightningTrainer
    reloaded_trainer = DeepChemLightningTrainer.load_checkpoint(
        checkpoint_path, 
        model=dc_hf_model_new,
        batch_size=2,
        accelerator="gpu",
        devices=-1,  # Use single device for consistency
        logger=False,
        enable_progress_bar=False,
        # precision="16-mixed",
    )
    state_reloaded = reloaded_trainer.lightning_model.pt_model.state_dict()
    
    # --- Correctness Check 2: After Loading ---
    # Verify that the reloaded state dict has the same keys
    assert state_after_training.keys() == state_reloaded.keys()
    
    # Verify that the reloaded weights are identical to the saved weights
    for key in state_after_training:
        torch.testing.assert_close(
            state_after_training[key].detach().cpu(),
            state_reloaded[key].detach().cpu(),
            msg=f"Weight mismatch for key {key} after reloading.",
            rtol=1e-4, atol=1e-6
        )
    
    # --- Correctness Check 3: Functional Equivalence ---
    # Predict with both models and compare results to ensure they are identical
    original_preds_batches = trainer.predict(dataset=dataset, num_workers=0)
    reloaded_preds_batches = reloaded_trainer.predict(dataset=dataset, num_workers=0)
    
    # Handle potential None values in prediction batches for FSDP
    if original_preds_batches is not None and reloaded_preds_batches is not None:
        original_valid = [p for p in original_preds_batches if p is not None]
        reloaded_valid = [p for p in reloaded_preds_batches if p is not None]
        
        if original_valid and reloaded_valid:
            original_preds = np.concatenate(original_valid)
            reloaded_preds = np.concatenate(reloaded_valid)
            
            np.testing.assert_allclose(
                original_preds,
                reloaded_preds,
                err_msg="Predictions from original and reloaded models do not match.",
                rtol=1e-4, atol=1e-6
            )


@pytest.mark.torch
def test_chemberta_overfit_with_lightning_trainer(smiles_data, chemberta_tokenizer):
    """
    Tests if the ChemBERTa model can overfit to a small dataset using Lightning trainer with FSDP.
    This validates that the Lightning training loop works correctly and the model
    can learn from the data by achieving good performance on the training set.
    Also tests the multi-GPU prediction capability with checkpoint saving and loading.
    """
    import torch
    import deepchem as dc
    L.seed_everything(42)
    
    dataset = smiles_data["dataset"]
    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"
    mae_metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
   
    classification_dataset = dataset
    
    # Create ChemBERTa model for classification
    dc_hf_model = Chemberta(
        task='regression',
        tokenizer_path=tokenizer_path,
        device='cpu',
        batch_size=1,  # Smaller batch size to ensure all samples are processed
        learning_rate=0.0005  # Slightly lower learning rate for better convergence
    )
    

    # Create Lightning trainer with optimized settings for multi-GPU HuggingFace models with FSDP
    lightning_trainer = DeepChemLightningTrainer(
        model=dc_hf_model,
        batch_size=1,  # Match model batch size to ensure all samples processed
        max_epochs=70,  # More epochs for overfitting
        accelerator="gpu",
        strategy="fsdp",
        devices=-1,  # Use all available GPUs
        logger=False,
        enable_progress_bar=False,
        precision="16-mixed",
    )
    

    eval_before = dc_hf_model.evaluate(
        dataset=classification_dataset,
        metrics=[dc.metrics.Metric(dc.metrics.mean_absolute_error)]
    )

    lightning_trainer.fit(classification_dataset, num_workers=0)
    
    # Save checkpoint after training
    lightning_trainer.save_checkpoint("chemberta_overfit_best.ckpt")
    
    
    new_dc_hf_model = Chemberta(
        task='regression',
        tokenizer_path=tokenizer_path,
        device='cpu',
        batch_size=1,  # Match model batch size
        learning_rate=0.0005
    )
    
    # Load the checkpoint into the new model instance
    reloaded_trainer = DeepChemLightningTrainer.load_checkpoint(
        "chemberta_overfit_best.ckpt", model=new_dc_hf_model,
        batch_size=1,
        accelerator="gpu",
        devices=1,  
        logger=False,
        enable_progress_bar=False,
    )

    # Evaluate the model on the training set
    eval_score = reloaded_trainer.evaluate(
        dataset=dataset,
        metrics=[mae_metric]
    )
    print(eval_before, eval_score)
    assert eval_before[mae_metric.name] > eval_score[mae_metric.name]*2, "Model did not overfit as expected"

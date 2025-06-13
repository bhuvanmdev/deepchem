import pytest
from deepchem.feat import WeaveFeaturizer
import deepchem as dc
import numpy as np
import torch
from deepchem.models.torch_models import WeaveModel
# from deepchem.models.tests.test_weavemodel_pytorch import get_dataset
# try:
from deepchem.models.lightning.trainer2 import DeepChemLightningTrainer
import lightning as L
# except ImportError as e:
#     print(f"DeepChem Lightning module not found: {e}")

@pytest.mark.pyl
def test_multitask_classifier():
    tasks, datasets, _ = dc.molnet.load_clintox()
    _, valid_dataset, _ = datasets
    print(f"Number of tasks: {len(tasks)} and number of samples: {len(valid_dataset)}")
    
    model = dc.models.MultitaskClassifier(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes=[1000],
        dropouts=0.2,
        learning_rate=0.0001,
        device="cpu",
        batch_size=16
    )
    
    trainer = DeepChemLightningTrainer(
        model=model,
        batch_size=16,
        max_epochs=30,
        accelerator="cuda",
        devices=-1,
        log_every_n_steps=1,
        strategy="fsdp",
        # fast_dev_run=True
    )
    
    trainer.fit(valid_dataset)
    trainer.save_checkpoint("multitask_classifier.ckpt")

    # Reload model and checkpoint
    model = dc.models.MultitaskClassifier(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes=[1000],
        dropouts=0.2,
        learning_rate=0.0001,
        device="cpu",
        batch_size=16
    )
    
    trainer = DeepChemLightningTrainer(
        model=model,
        batch_size=16,
        max_epochs=10,
        accelerator="cuda",
        devices=-1,
        log_every_n_steps=1,
        strategy="fsdp",
    )
    
    trainer.load_checkpoint("multitask_classifier.ckpt")
    predictions = trainer.predict(valid_dataset)
    print(predictions[0])

@pytest.mark.pyl
def test_gcn_model():
    featurizer = dc.feat.MolGraphConvFeaturizer()
    tasks, all_dataset, transformers = dc.molnet.load_bace_classification(featurizer)
    train_dataset, valid_dataset, test_dataset = all_dataset
    print(f"Number of tasks: {len(tasks)} and number of samples: {len(valid_dataset)}")
    
    model = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        batch_size=16,
        learning_rate=0.001,
        device="cpu"
    )
    
    trainer = DeepChemLightningTrainer(
        model=model,
        batch_size=16,
        max_epochs=10,
        accelerator="cuda",
        devices=-1,
        log_every_n_steps=1,
        strategy="fsdp",
        fast_dev_run=True
    )
    
    trainer.fit(valid_dataset)
    trainer.save_checkpoint("gcn_model.ckpt")

    # Reload model and checkpoint
    model = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        batch_size=16,
        learning_rate=0.001,
        device="cpu"
    )
    
    trainer = DeepChemLightningTrainer(
        model=model,
        batch_size=16,
        max_epochs=10,
        accelerator="cuda",
        devices=-1,
        log_every_n_steps=1,
        strategy="fsdp",
    )
    
    trainer.load_checkpoint("gcn_model.ckpt")
    predictions = trainer.predict(valid_dataset)
    print(predictions[0])

@pytest.mark.pyl
def load_hyperparam_from_ckpt():
    tasks, datasets, _ = dc.molnet.load_clintox()
    _, valid_dataset, _ = datasets
    print(f"Number of tasks: {len(tasks)} and number of samples: {len(valid_dataset)}")
    
    model = dc.models.MultitaskClassifier(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes=[1000],
        dropouts=0.2,
        learning_rate=0.0001,
        device="cpu",
        batch_size=16
    )
    
    trainer = DeepChemLightningTrainer(
        model=model,
        batch_size=16,
        max_epochs=1,
        accelerator="cuda",
        devices=-1,
        log_every_n_steps=1,
        strategy="fsdp",
        # fast_dev_run=True
    )
    
    trainer.fit(valid_dataset)
    trainer.save_checkpoint("multitask_classifier.ckpt")

    # Reload model and checkpoint
    model = dc.models.MultitaskClassifier(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes=[1000],
        dropouts=0.2,
        learning_rate=0.001,
        device="cpu",
        batch_size=16
    )
    trainer = DeepChemLightningTrainer(
        model=model,
        batch_size=16,
        max_epochs=1,
        accelerator="cuda",
        devices=-1,
        log_every_n_steps=1,
        strategy="fsdp",
    )
    trainer.load_checkpoint("multitask_classifier.ckpt")
    assert 0.0001 == trainer.model.learning_rate, "Learning rate should be 0.0001"
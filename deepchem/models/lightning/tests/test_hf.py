
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from deepchem.molnet import load_zinc15
from deepchem.feat import RobertaFeaturizer, DummyFeaturizer
from deepchem.data import CSVLoader, DiskDataset
from prepare_data import *


## used for data loading (run only once to generate the data and featurize it)
# generate_deepchem_splits(dataset_names=["clintox"],
#                              output_dir=r"E:\dc\deepchem_data\raw",
#                              clean_smiles=True,
#                              max_smiles_len=200)

# featurize_datasets(dataset_names=["clintox"],
#                    featurizer_names=["dummy"],
#                    data_root=r"E:\dc\deepchem_data\raw",
#                    save_root=r"E:\dc\deepchem_data\featurized",)

def smiles_regression_dataset():
    hf_tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/PubChem10M_SMILES_BPE_60k")

    data_root = r"E:\dc\deepchem_data\featurized"
    dataset_name = "clintox"
    featurizer_name = "dummy_featurized"
    dataset_path = os.path.join(data_root,featurizer_name, dataset_name)
    
    # Load train, validation, and test datasets
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid") 
    test_path = os.path.join(dataset_path, "test")

    # Load the dataset from path
    

    config = RobertaConfig(vocab_size=hf_tokenizer.vocab_size)
    model = RobertaForMaskedLM(config)

    hf_model = HuggingFaceModel(model=model,
                                tokenizer=hf_tokenizer,
                                task='mlm',
                                device='cuda')
    loss = hf_model.fit(DiskDataset(valid_path), nb_epoch=1)

    return loss

print(smiles_regression_dataset())
import os
import argparse
import deepchem as dc
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict


# Define supported featurizers here
FEATURIZER_DICT = {
    "dmpnn": dc.feat.DMPNNFeaturizer(),
    "dummy": dc.feat.DummyFeaturizer(),
    "grover": dc.feat.GroverFeaturizer(features_generator=dc.feat.CircularFingerprint()),
    "ecfp": dc.feat.CircularFingerprint(size=1024),
    "molgraphconv": dc.feat.MolGraphConvFeaturizer(use_edges=True),
    "rdkit_conformer": dc.feat.RDKitConformerFeaturizer(),
}

task_dict = {'bbbp': ['p_np'], 
            'bace_classification': ['Class'],
            'clintox': ['FDA_APPROVED', 'CT_TOX'],
            'hiv': ['HIV_active'],
            'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
            'sider': [
                'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
                'Product issues', 'Eye disorders', 'Investigations',
                'Musculoskeletal and connective tissue disorders',
                'Gastrointestinal disorders', 'Social circumstances',
                'Immune system disorders', 'Reproductive system and breast disorders',
                'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                'General disorders and administration site conditions',
                'Endocrine disorders', 'Surgical and medical procedures',
                'Vascular disorders', 'Blood and lymphatic system disorders',
                'Skin and subcutaneous tissue disorders',
                'Congenital, familial and genetic disorders', 'Infections and infestations',
                'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
                'Renal and urinary disorders',
                'Pregnancy, puerperium and perinatal conditions',
                'Ear and labyrinth disorders', 'Cardiac disorders',
                'Nervous system disorders', 'Injury, poisoning and procedural complications'
            ],
            # esol is an alias for delaney dataset
            'delaney': ['measured log solubility in mols per litre'],
            'freesolv': ['y'],
            'lipo': ['exp'],
            'clearance': ['target'],
            'bace_regression': ['pIC50']
            }


def generate_deepchem_splits(dataset_names: List, 
                             output_dir: str, 
                             clean_smiles: bool=True, 
                             max_smiles_len: int=200):
    """
    Generates scaffold splits for DeepChem datasets, optionally cleans them, and saves as CSVs.

    Parameters
    ----------
    dataset_names: List(str)
        List of dataset names.
    output_dir: str
        Folder to save the CSVs.
    clean_smiles: bool
        Whether to filter SMILES strings by max length.
    max_smiles_len: int
        Maximum allowed SMILES length if cleaning is enabled.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"deepchem_split_log_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

    logging.info("Starting DeepChem dataset splitting...")

    for dataset_name in dataset_names:
        try:
            logging.info(f"Processing dataset: {dataset_name}")
            load_fn = getattr(dc.molnet, f'load_{dataset_name}')

            task_names, (train_set, valid_set, test_set), transformers = load_fn(
                featurizer=dc.feat.DummyFeaturizer(),
                transformers=[],
                splitter='scaffold',
                reload=False
            )

            dataset_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            split_names = ['train', 'valid', 'test']
            datasets = [train_set, valid_set, test_set]

            for split_name, split_set in zip(split_names, datasets):
                smiles_list = split_set.X
                y = split_set.y

                if clean_smiles:
                    mask = pd.Series(smiles_list).str.len() <= max_smiles_len
                    removed_count = (~mask).sum()
                    logging.info(f"{dataset_name} {split_name}: removed {removed_count} rows (SMILES > {max_smiles_len})")
                    smiles_list = [s for i, s in enumerate(smiles_list) if mask.iloc[i]]
                    y = y[mask.to_numpy()]

                data = {'smiles': smiles_list}
                data.update({task_names[i]: y[:, i] for i in range(len(task_names))})
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(dataset_dir, f"{split_name}.csv"), index=False)
                logging.info(f"Saved {split_name} split to {dataset_dir}/{split_name}.csv")

        except AttributeError:
            logging.error(f"Dataset '{dataset_name}' not found in DeepChem.molnet. Skipping.")
        except Exception as e:
            logging.exception(f"Error while processing {dataset_name}: {str(e)}")

    logging.info("All datasets processed.")


def clean_smiles_by_length(df, smiles_column="smiles", max_length=200):
    """
    Removes rows where the SMILES string exceeds `max_length`.
    """
    return df[df[smiles_column].str.len() <= max_length].reset_index(drop=True)


def featurize_datasets(
    dataset_names: List,
    featurizer_names: List,
    data_root: str,
    save_root: str,
    log_dir: str=None,
    smiles_column: str="smiles"
):
    """
    Featurizes datasets using selected DeepChem featurizers by name.

    Parameters
    ----------
    dataset_names: List
        List of dataset names.
    task_dict: Dict
        Mapping from dataset name -> list of task names.
    featurizer_names: List
        List of featurizer keys from FEATURIZER_DICT.
    data_root: str
        Path to cleaned CSVs.
    save_root: str
        Path to save featurized datasets.
    splits: List
        Dataset splits to process.
    log_dir: str
        Path for logs; defaults to save_root.
    smiles_column: str
        Name of the SMILES column.
    """

    log_dir = log_dir or save_root
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join(log_dir, f"featurization_log_{timestamp}.log")

    logger = logging.getLogger('featurize')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())  # console log

    logger.info("Starting featurization.")
    logger.info(f"Datasets: {dataset_names}")
    logger.info(f"Featurizers: {featurizer_names}")

    for featurizer_name in featurizer_names:
        if featurizer_name not in FEATURIZER_DICT:
            logger.warning(f"Featurizer {featurizer_name} not found in FEATURIZER_DICT. Skipping.")
            continue

        featurizer = FEATURIZER_DICT[featurizer_name]

        for dataset in dataset_names:
            if dataset not in task_dict:
                logger.warning(f"Skipping {dataset} - missing in task_dict.")
                continue

            for split in ["train", "valid", "test"]:
                input_csv = os.path.join(data_root, dataset, f"{split}.csv")
                cleaned_input_csv = os.path.join(data_root, dataset, f"{split}_cleaned.csv")
                if not os.path.exists(input_csv):
                    logger.warning(f"File not found: {input_csv}")
                    continue

                try:
                    df = pd.read_csv(input_csv)
                    original_len = len(df)
                    df = clean_smiles_by_length(df, smiles_column=smiles_column)
                    cleaned_len = len(df)

                    if cleaned_len == 0:
                        logger.warning(f"All SMILES were removed after cleaning for {input_csv}. Skipping.")
                        continue

                    df.to_csv(cleaned_input_csv, index=False)

                    logger.info(f"{dataset}-{split}: {original_len} → {cleaned_len} rows after length-based cleaning")

                    logger.info(f"Featurizing: {dataset}-{split} using {featurizer_name}")
                    loader = dc.data.CSVLoader(
                        tasks=task_dict[dataset],
                        feature_field=smiles_column,
                        featurizer=featurizer
                    )

                    featurized_dir = os.path.join(save_root, f'{featurizer_name}_featurized', dataset, split)
                    os.makedirs(featurized_dir, exist_ok=True)

                    dataset_obj = loader.create_dataset(
                        cleaned_input_csv, data_dir=featurized_dir
                    )

                    logger.info(f"Done: {dataset}-{split} with {featurizer_name} ({len(dataset_obj)} samples)")
                except Exception as e:
                    logger.exception(f"Error: {dataset}-{split} with {featurizer_name}: {e}")

    logger.info("Featurization process completed.")


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--split_type',
                        type=str,
                        help='type of the splits to use for the datasets : molformer/deepchem',
                        default='molformer')
    argparser.add_argument(
        '--datasets',
        type=str,
        help='comma-separated list of datasets to featurize',
        default='bbbp,bace,clintox,hiv,tox21,sider')
    argparser.add_argument('--featurizers',
                           type=str,
                           help='comma-separated list of featurizers to featurize the datasets')
    argparser.add_argument('--data_dir',
                           type=str,
                           help='data dir of stored splits in csv format')
    argparser.add_argument('--feat_dir',
                           type=str,
                           help='dir to store the featurized datasets')


    args = argparser.parse_args()

    datasets = args.datasets.split(',')
    featurizers = args.featurizers.split(',')

    if datasets is None:
        raise ValueError("Please provide a list of datasets to benchmark.")
    if not isinstance(datasets, list):
        raise ValueError("Datasets should be provided as a list.")
    if len(datasets) == 0:
        raise ValueError(
            "The list of datasets is empty. Please provide at least one dataset."
        )
    if args.split_type == 'deepchem':
        generate_deepchem_splits(dataset_names=datasets,
                                 output_dir=args.data_dir,
                                 clean_smiles=True,
                                 max_smiles_len=200)

    featurize_datasets(dataset_names=datasets,
                       featurizer_names=featurizers,
                       data_root=args.data_dir,
                       save_root=args.feat_dir,
                       smiles_column='smiles')

if __name__ == "__main__":
    main()

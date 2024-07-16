from immutabledict import immutabledict


# data_type_list
ALL_TASKS: tuple[str] = tuple(
    "Single AA Sequence",
    "Single SA Sequence",
    "Single UniProt ID",
    "Single PDB/CIF Structure",
    "Multiple AA Sequences",
    "Multiple SA Sequences",
    "Multiple UniProt IDs",
    "Multiple PDB/CIF Structures",
    "SaprotHub Dataset",
    "A pair of AA Sequences",
    "A pair of SA Sequences",
    "A pair of UniProt IDs",
    "A pair of PDB/CIF Structures",
    "Multiple pairs of AA Sequences",
    "Multiple pairs of SA Sequences",
    "Multiple pairs of UniProt IDs",
    "Multiple pairs of PDB/CIF Structures",
)

# task_type_dict
TASK_TYPE: immutabledict[str,str]  = immutabledict(
    {
        "Classify protein sequences (classification)": "classification",
        "Classify each Amino Acid (amino acid classification), e.g. Binding site detection": "token_classification",
        "Predict protein fitness (regression), e.g. Predict the Thermostability of a protein": "regression",
        "Predict protein-protein interaction (pair classification)": "pair_classification",
        "Predict protein-protein interaction (pair regression)": "pair_regression",
    }
)
# model_type_dict
TASK2MODEL: immutabledict[str,str] = immutabledict(
    {
        "classification": "saprot/saprot_classification_model",
        "token_classification": "saprot/saprot_token_classification_model",
        "regression": "saprot/saprot_regression_model",
        "pair_classification": "saprot/saprot_pair_classification_model",
        "pair_regression": "saprot/saprot_pair_regression_model",
    }
)


# dataset_type_dict
TASK2DATASET: immutabledict[str,str]  = immutabledict(
    {
        "classification": "saprot/saprot_classification_dataset",
        "token_classification": "saprot/saprot_token_classification_dataset",
        "regression": "saprot/saprot_regression_dataset",
        "pair_classification": "saprot/saprot_pair_classification_dataset",
        "pair_regression": "saprot/saprot_pair_regression_dataset",
    }
)

# training_data_type_dict
TRAINING_SEQUENCE_TYPE_MAPPING: immutabledict[str,str]  = immutabledict(
    {
        "Single AA Sequence": "AA",
        "Single SA Sequence": "SA",
        "Single UniProt ID": "SA",
        "Single PDB/CIF Structure": "SA",
        "Multiple AA Sequences": "AA",
        "Multiple SA Sequences": "SA",
        "Multiple UniProt IDs": "SA",
        "Multiple PDB/CIF Structures": "SA",
        "SaprotHub Dataset": "SA",
        "A pair of AA Sequences": "AA",
        "A pair of SA Sequences": "SA",
        "A pair of UniProt IDs": "SA",
        "A pair of PDB/CIF Structures": "SA",
        "Multiple pairs of AA Sequences": "AA",
        "Multiple pairs of SA Sequences": "SA",
        "Multiple pairs of UniProt IDs": "SA",
        "Multiple pairs of PDB/CIF Structures": "SA",
    }
)

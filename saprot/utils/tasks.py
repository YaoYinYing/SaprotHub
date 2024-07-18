from typing import Literal
from immutabledict import immutabledict


# task_type_dict, reversed
TASK_TYPE: immutabledict[str, str] = immutabledict(
    {
        "classification": "Classify protein sequences (classification)",
        "token_classification": "Classify each Amino Acid (amino acid classification), e.g. Binding site detection",
        "regression": "Predict protein fitness (regression), e.g. Predict the Thermostability of a protein",
        "pair_classification": "Predict protein-protein interaction (pair classification)",
        "pair_regression": "Predict protein-protein interaction (pair regression)",
    }
)
# model_type_dict
TASK2MODEL: immutabledict[str, str] = immutabledict(
    {
        "classification": "saprot/saprot_classification_model",
        "token_classification": "saprot/saprot_token_classification_model",
        "regression": "saprot/saprot_regression_model",
        "pair_classification": "saprot/saprot_pair_classification_model",
        "pair_regression": "saprot/saprot_pair_regression_model",
    }
)


# dataset_type_dict
TASK2DATASET: immutabledict[str, str] = immutabledict(
    {
        "classification": "saprot/saprot_classification_dataset",
        "token_classification": "saprot/saprot_token_classification_dataset",
        "regression": "saprot/saprot_regression_dataset",
        "pair_classification": "saprot/saprot_pair_classification_dataset",
        "pair_regression": "saprot/saprot_pair_regression_dataset",
    }
)


# data_type_list
ALL_TASKS: tuple[str] = tuple(TASK2MODEL.keys())
ALL_TASKS_HINT= Literal['classification','token_classification','regression','pair_classification','pair_regression']
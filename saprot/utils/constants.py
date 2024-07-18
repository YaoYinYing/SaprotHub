import itertools
from typing import Literal

from immutabledict import immutabledict


aa_set = {
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
}
aa_list = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

struc_unit = "abcdefghijklmnopqrstuvwxyz"


DATA_TYPES: tuple = (
    "Single_AA_Sequence",
    "Single_SA_Sequence",
    "Single_UniProt_ID",
    "Single_PDB/CIF_Structure",
    "Multiple_AA_Sequences",
    "Multiple_SA_Sequences",
    "Multiple_UniProt_IDs",
    "Multiple_PDB/CIF_Structures",
    "SaprotHub_Dataset",
    "A_pair_of_AA_Sequences",
    "A_pair_of_SA_Sequences",
    "A_pair_of_UniProt_IDs",
    "A_pair_of_PDB/CIF_Structures",
    "Multiple_pairs_of_AA_Sequences",
    "Multiple_pairs_of_SA_Sequences",
    "Multiple_pairs_of_UniProt_IDs",
    "Multiple_pairs_of_PDB/CIF_Structures",
)

DATA_TYPES_HINT = Literal[
    "Single_AA_Sequence",
    "Single_SA_Sequence",
    "Single_UniProt_ID",
    "Single_PDB/CIF_Structure",
    "Multiple_AA_Sequences",
    "Multiple_SA_Sequences",
    "Multiple_UniProt_IDs",
    "Multiple_PDB/CIF_Structures",
    "SaprotHub_Dataset",
    "A_pair_of_AA_Sequences",
    "A_pair_of_SA_Sequences",
    "A_pair_of_UniProt_IDs",
    "A_pair_of_PDB/CIF_Structures",
    "Multiple_pairs_of_AA_Sequences",
    "Multiple_pairs_of_SA_Sequences",
    "Multiple_pairs_of_UniProt_IDs",
    "Multiple_pairs_of_PDB/CIF_Structures",
]


# training_data_type_dict
DATASET_TYPE2SEQ_TYPE: immutabledict[str, str] = immutabledict(
    {
        "Single_AA_Sequence": "AA",
        "Single_SA_Sequence": "SA",
        "Single_UniProt_ID": "SA",
        "Single_PDB/CIF_Structure": "SA",
        "Multiple_AA_Sequences": "AA",
        "Multiple_SA_Sequences": "SA",
        "Multiple_UniProt_IDs": "SA",
        "Multiple_PDB/CIF_Structures": "SA",
        "SaprotHub_Dataset": "SA",
        "A_pair_of_AA_Sequences": "AA",
        "A_pair_of_SA_Sequences": "SA",
        "A_pair_of_UniProt_IDs": "SA",
        "A_pair_of_PDB/CIF_Structures": "SA",
        "Multiple_pairs_of_AA_Sequences": "AA",
        "Multiple_pairs_of_SA_Sequences": "SA",
        "Multiple_pairs_of_UniProt_IDs": "SA",
        "Multiple_pairs_of_PDB/CIF_Structures": "SA",
    }
)


def create_vocab(size: int) -> dict:
    """

    Args:
        size:   Size of the vocabulary

    Returns:
        vocab:  Vocabulary
    """

    token_len = 1
    while size > len(struc_unit) ** token_len:
        token_len += 1

    vocab = {}
    for i, token in enumerate(itertools.product(struc_unit, repeat=token_len)):
        vocab[i] = "".join(token)
        if len(vocab) == size:
            vocab[i + 1] = "#"
            return vocab

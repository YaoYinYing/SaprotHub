import os
from typing import Tuple

import torch

from saprot.utils.foldseek_util import (
    FoldSeekSetup,
    Mask,
    get_struc_seq,
    StrucuralSequence,
)
from saprot.utils.weights import PretrainedModel, SaProtModelHint
from saprot.model.saprot.saprot_if_model import SaProtIFModel, IF_METHOD_HINT


def inverse_folding(
    input_structure: str, chain_id: str, mask_area: str, save_dir: str = "."
) -> Tuple[StrucuralSequence, list[str], Mask]:
    model_loader = PretrainedModel(
        dir=os.path.abspath("./weights/SaProt/"),
        model_name="SaProt_650M_AF2_inverse_folding",
    )

    foldseek = FoldSeekSetup(
        bin_dir="./bin",
        base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
    ).foldseek

    # pdb_path = "example/1ubq.cif"
    pdb_path = input_structure

    print(foldseek)
    parsed_seqs = get_struc_seq(foldseek, pdb_path, [chain_id], plddt_mask=False)[
        chain_id
    ]
    print(parsed_seqs)

    # `method` refers to the prediction method. It could be either "argmax" or "multinomial".
    # - `argmax` selects the amino acid with the highest probability.
    # - `multinomial` samples an amino acid from the multinomial distribution.

    method: IF_METHOD_HINT = "multinomial"
    num_samples = 10  # @param {type:"integer"}

    mask_string = mask_area

    mask = Mask(mask_string)

    # @markdown - `num_samples` refers to the number of output amino acid sequences.

    save_name = "predicted_seq"  # @param {type:"string"}

    masked_aa_seq = parsed_seqs.masked_seq(mask)
    masked_struc_seq = parsed_seqs.struc_seq

    print(f"{mask=}")
    print(f"{masked_aa_seq=}, {masked_struc_seq=}")

    # assert len(masked_aa_seq) == len(masked_struc_seq), f"Please make sure that the amino acid sequence ({len(masked_aa_seq)}) and the structure sequence ({len(masked_struc_seq)}) have the same length."
    # masked_sa_seq = ''.join(a + b for a, b in zip(masked_aa_seq, masked_struc_seq))

    config = {
        "config_path": model_loader.weights_dir,
        "load_pretrained": True,
    }

    saprot_if_model = SaProtIFModel(**config)
    tokenizer = saprot_if_model.tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saprot_if_model.to(device)

    ################################################################################
    ############################### Predict ########################################
    ################################################################################

    pred_aa_seqs = saprot_if_model.predict(
        masked_aa_seq, masked_struc_seq, method=method, num_samples=num_samples
    )

    print("#" * 100)

    save_path = os.path.join(save_dir, f"{save_name}.fasta")

    with open(save_path, "w") as w:
        w.write(f">{parsed_seqs.name}_{parsed_seqs.chain}\n{parsed_seqs.seq}\n")
        for i, aa_seq in enumerate(pred_aa_seqs):
            print(aa_seq)
            w.write(f">>{parsed_seqs.name}_{parsed_seqs.chain}_pred_{i}\n{aa_seq}\n")

    new_mask = Mask(None)
    new_mask.update_from_masked_sequence(masked_aa_seq)
    print(new_mask)

    return parsed_seqs, pred_aa_seqs, mask


def main():

    pdb_path = "example/1ubq.cif"
    mask_string = "1-75"
    chain_id = "A"

    save_dir = f"output/inverse_folding/"
    os.makedirs(save_dir, exist_ok=True)

    parsed_seqs, pred_aa_seqs, mask = inverse_folding(
        input_structure=pdb_path,
        chain_id=chain_id,
        mask_area=mask_string,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()

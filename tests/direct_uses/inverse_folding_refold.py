import os

import torch

from saprot.utils.foldseek_util import (
    Mask,
    StructuralAwareSequence,
)
from saprot.utils.weights import PretrainedModel

from saprot.utils.data_preprocess import convert_outputs_to_pdb


from .inverse_folding import inverse_folding


def refold(
    structure_seq: StructuralAwareSequence,
    seqs: list[str],
    mask: Mask,
    save_dir: str = ".",
):
    model_loader = PretrainedModel(
        dir=os.path.abspath("./weights/SaProt/"),
        model_name="esmfold_v1",
        huggingface_id="facebook",
        loader_type="esmfold",
    )

    esmfold, tokenizer = model_loader.load_model()
    esmfold.esm = esmfold.esm.half()
    esmfold.trunk.set_chunk_size(64)

    device = esmfold.device

    fold_id = structure_seq.name
    chain_id = structure_seq.chain
    mask_id = mask.mask_pos_range

    for i, seq in enumerate(seqs):
        run_esmfold(
            model=esmfold,
            tokenizer=tokenizer,
            save_name=f"{fold_id}_{chain_id}_{mask_id}_pred_{i}",
            seq=seq,
            save_dir=save_dir,
        )


def run_esmfold(model, tokenizer, save_name, seq, save_dir):
    ################################################################################
    ################################## PREDICT ###################################
    ################################################################################
    tokenized_input = tokenizer(
        [seq],
        return_tensors="pt",
        add_special_tokens=False,
        max_length=1024,
        truncation=True,
    )["input_ids"]

    tokenized_input = tokenized_input.to(model.device)
    with torch.no_grad():
        output = model(tokenized_input)

        ################################################################################
        #################################### SAVE ####################################
        ################################################################################
        save_path = os.path.join(save_dir, f"{save_name}.pdb")
        pdb = convert_outputs_to_pdb(output)
        with open(save_path, "w") as f:
            f.write("".join(pdb))

        print("Predicted structure")


def main():
    pdb_path = "example/1ubq.cif"
    mask_string = "1-75"
    chain_id = "A"

    save_path = "output/inverse_folding_refold/"
    os.makedirs(save_path, exist_ok=True)

    parsed_seqs, pred_aa_seqs, mask = inverse_folding(
        input_structure=pdb_path,
        mask_area=mask_string,
        chain_id=chain_id,
        save_dir=save_path,
    )

    pdb_save_path = os.path.join(save_path, "esmfold")
    os.makedirs(pdb_save_path, exist_ok=True)
    refold(
        structure_seq=parsed_seqs,
        seqs=pred_aa_seqs,
        mask=mask,
        save_dir=pdb_save_path,
    )


if __name__ == "__main__":
    main()

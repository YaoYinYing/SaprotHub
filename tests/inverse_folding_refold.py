import os

import torch

from saprot.utils.foldseek_util import (
    Mask,
    StrucuralSequence,
)
from saprot.utils.weights import PretrainedModel


from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


from inverse_folding import inverse_folding


def refold(
    structure_seq: StrucuralSequence, seqs: list[str], mask: Mask, save_dir: str = "."
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    esmfold.to(device)

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


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    outputs["plddt"] *= 100

    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


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
        structure_seq=parsed_seqs, seqs=pred_aa_seqs, mask=mask, save_dir=pdb_save_path
    )


if __name__ == "__main__":
    main()

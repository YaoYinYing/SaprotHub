import os
import pandas as pd
import torch

from saprot.utils.foldseek_util import StrucuralSequences, get_struc_seq
from saprot.model.saprot.saprot_foldseek_mutation_model import (
    SaprotFoldseekMutationModel,
)
from saprot.utils.foldseek_util import FoldSeekSetup
from saprot.utils.weights import PretrainedModel


def get_model():
    model_loader = PretrainedModel(
        dir=os.path.abspath("./weights/SaProt/"),
        model_name="SaProt_650M_AF2",
    )

    foldseek = FoldSeekSetup(
        bin_dir="./bin",
        base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
    ).foldseek

    config = {
        "foldseek_path": foldseek,
        "config_path": model_loader.weights_dir,
        "load_pretrained": True,
    }
    model = SaprotFoldseekMutationModel(**config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    return model


def run_dms(
    model: SaprotFoldseekMutationModel, parsed_seqs: StrucuralSequences, chain_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run DMS on the mutant sequence.
    """

    PSSM_Alphabet = "ARNDCQEGHILKMFPSTWYV"
    wt_seq = parsed_seqs.get(chain_id, None)
    if wt_seq is None:
        raise ValueError(f"No sequence found for chain {chain_id}")

    all_dict_aa = []
    all_dict_prob = []

    print(f"Running DMS against chain {chain_id}")

    for resi, resn in enumerate(wt_seq.seq):
        print(f"Running DMS on {resi+1}, {resn=}")

        mut_dict = model.predict_pos_mut(seq=wt_seq.combined_sequence, pos=resi+1)
        #print(f'{mut_dict=}')
        mut_dict_prob = model.predict_pos_prob(seq=wt_seq.combined_sequence, pos=resi+1)
        #print(f'{mut_dict_prob=}')
        # sort this dict by PSSM_Alphabet
        sorted_df_aa = {aa: mut_dict[f"{resn}{resi+1}{aa}"] for aa in PSSM_Alphabet}
        sorted_df_aa_prob = {
            aa: mut_dict_prob[aa] for aa in PSSM_Alphabet
        }
        all_dict_aa.append(sorted_df_aa)
        all_dict_prob.append(sorted_df_aa_prob)



    df_aa_score = pd.DataFrame(all_dict_aa)
    df_aa_score_prob = pd.DataFrame(all_dict_prob)

    return df_aa_score, df_aa_score_prob


def main():
    pdb_path = "example/1ubq.cif"
    chain_id = "A"

    output_dir = "output/deep_mutagenese_scan"
    os.makedirs(output_dir, exist_ok=True)

    model = get_model()
    foldseek = model.foldseek_path

    print(foldseek)
    parsed_seqs = get_struc_seq(foldseek, pdb_path, chains=[chain_id], plddt_mask=False)

    print(f"{parsed_seqs=}")

    df_aa_score, df_aa_score_prob = run_dms(
        model=model, parsed_seqs=parsed_seqs, chain_id=chain_id
    )

    df_aa_score.to_csv(os.path.join(output_dir, "aa_score.csv"))
    df_aa_score_prob.to_csv(os.path.join(output_dir, "aa_score_prob.csv"))


if __name__ == "__main__":
    main()

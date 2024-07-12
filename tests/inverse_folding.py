import os

from saprot.utils.foldseek_util import FoldSeekSetup, Mask, get_struc_seq
from saprot.utils.weights import PretrainedModel, SaProtModelHint
from saprot.model.saprot.saprot_if_model import SaProtIFModel,IF_METHOD_HINT

model_loader = PretrainedModel(
    dir=os.path.abspath("./weights/SaProt/"),
    model_name="SaProt_650M_AF2_inverse_folding",
)

foldseek = FoldSeekSetup(
    bin_dir="./bin",
    base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
).foldseek

pdb_path = "example/1ubq.cif"
#pdb_path = "example/8ac8.cif"

print(foldseek)
parsed_seqs = get_struc_seq(foldseek, pdb_path, ["A"], plddt_mask=False)["A"]
print(parsed_seqs)


#`method` refers to the prediction method. It could be either "argmax" or "multinomial".
# - `argmax` selects the amino acid with the highest probability.
# - `multinomial` samples an amino acid from the multinomial distribution.

method:IF_METHOD_HINT = "multinomial" 
num_samples = 10 # @param {type:"integer"}

mask_string='1-75'

mask=Mask(mask_string)

#@markdown - `num_samples` refers to the number of output amino acid sequences.

save_name = "predicted_seq" # @param {type:"string"}

masked_aa_seq = parsed_seqs.masked_seq(mask)
masked_struc_seq = parsed_seqs.struc_seq

print(f'{mask=}')
print(f'{masked_aa_seq=}, {masked_struc_seq=}')

# assert len(masked_aa_seq) == len(masked_struc_seq), f"Please make sure that the amino acid sequence ({len(masked_aa_seq)}) and the structure sequence ({len(masked_struc_seq)}) have the same length."
# masked_sa_seq = ''.join(a + b for a, b in zip(masked_aa_seq, masked_struc_seq))


config = {
    "config_path": model_loader.weights_dir,
    "load_pretrained": True,
}


saprot_if_model = SaProtIFModel(**config)
tokenizer = saprot_if_model.tokenizer
device = "cpu"
saprot_if_model.to(device)



################################################################################
############################### Predict ########################################
################################################################################

pred_aa_seqs = saprot_if_model.predict(masked_aa_seq, masked_struc_seq, method=method, num_samples=num_samples)

print("#"*100)

save_path = f"output/inverse_folding/{save_name}.fasta"
os.makedirs(os.path.dirname(save_path),exist_ok=True)



with open(save_path, "w") as w:
    w.write(f">{parsed_seqs.name}_{parsed_seqs.chain}\n{parsed_seqs.seq}\n")
    for i, aa_seq in enumerate(pred_aa_seqs):
        print(aa_seq)
        w.write(f">>{parsed_seqs.name}_{parsed_seqs.chain}_pred_{i}\n{aa_seq}\n")


new_mask=Mask(None)
new_mask.update_from_masked_sequence(masked_aa_seq)
print(new_mask)
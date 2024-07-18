import os

import pandas as pd
import torch
from saprot.utils.data_preprocess import InputDataDispatcher
from saprot.utils.foldseek_util import FoldSeekSetup,FoldSeek
from saprot.utils.weights import AdaptedModel

import torch.nn.functional as F

subcellular_table = (
    "Nucleus",
    "Cytoplasm",
    "Extracellular",
    "Mitochondrion",
    "Cell.membrane",
    "Endoplasmic.reticulum",
    "Plastid",
    "Golgi.apparatus",
    "Lysosome/Vacuole",
    "Peroxisome",
)

foldseek = FoldSeekSetup(
    bin_dir="./bin",
    base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
).foldseek

dispatcher=InputDataDispatcher(
    DATASET_HOME="output/tasks/token_classification/dataset",
    LMDB_HOME="output/tasks/token_classification/lmdb",
    STRUCTURE_HOME="output/tasks/token_classification/structures/",
    FOLDSEEK_PATH=foldseek,
    nproc=20,
)


def get_subcellular_model():
    weight_worker = AdaptedModel(
        dir="./weights/SaProt/",
        huggingface_id="SaProtHub",
        model_name="Model-Binding_Site_Detection-650M",
        task_type="token_classification",
        num_of_categories=10
    )

    seqs=dispatcher.parse_data("Multiple_AA_Sequences", 'https://raw.githubusercontent.com/westlake-repl/SaprotHub/main/upload_files/%5BEXAMPLE%5D%5BAminoAcidClassification-3Categories%5DMultiple_AA_Sequences.csv')

    model,tokenizer=weight_worker.load_model()


    outputs_list=[]

    for i, s in enumerate(seqs):
        inputs = tokenizer(s.amino_acid_seq, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad(): 
            outputs = model(inputs)
        outputs_list.append(outputs)

    softmax_output_list = [F.softmax(output, dim=1).squeeze().tolist() for output in outputs_list]

    print("The probability of each category:")
    for seq_index, seq in enumerate(softmax_output_list):
        seq_prob_df = pd.DataFrame(seq)
        print('='*100)
        print(f'Sequence {seq_index + 1}:')
        print(seq_prob_df[1:-1].to_string())




def main():

    

    get_subcellular_model()


if __name__ == "__main__":
    main()

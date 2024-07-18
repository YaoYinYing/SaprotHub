import os

import torch
from saprot.utils.data_preprocess import InputDataDispatcher
from saprot.utils.foldseek_util import FoldSeekSetup,FoldSeek
from saprot.utils.weights import AdaptedModel

import torch.nn.functional as F


foldseek = FoldSeekSetup(
    bin_dir="./bin",
    base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
).foldseek

dispatcher=InputDataDispatcher(
    DATASET_HOME="output/tasks/classification/dataset",
    LMDB_HOME="output/tasks/classification/lmdb",
    STRUCTURE_HOME="output/tasks/classification/structures/",
    FOLDSEEK_PATH=foldseek,
    nproc=20,
)


def get_subcellular_model():
    weight_worker = AdaptedModel(
        dir="./weights/SaProt/",
        huggingface_id="SaProtHub",
        model_name="Model-Subcellular_Localization-650M",
        task_type="classification",
        num_of_categories=10
    )

    seqs=dispatcher.parse_data("Multiple_SA_Sequences", 'upload_files/[EXAMPLE]Multiple_SA_Sequences.csv')


    model,tokenizer=weight_worker.load_model()


    outputs_list=[]

    for i, s in enumerate(seqs):
        inputs = tokenizer(s.combined_sequence, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad(): 
            outputs = model(inputs)
        outputs_list.append(outputs)

    softmax_output_list = [F.softmax(output, dim=1).squeeze().tolist() for output in outputs_list]

    for index, output in enumerate(softmax_output_list):
        print(f"For Sequence {index}, Prediction: Category {output.index(max(output))}, Probability: {output}")
        






def main():

    

    get_subcellular_model()


if __name__ == "__main__":
    main()

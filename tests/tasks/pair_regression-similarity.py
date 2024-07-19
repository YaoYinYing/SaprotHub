import os

import torch
from rich.progress import track
from saprot.utils.data_preprocess import (
    InputDataDispatcher,
    StructuralAwareSequencePair,
)
from saprot.utils.foldseek_util import FoldSeekSetup
from saprot.utils.middleware import SADataAdapter
from saprot.utils.weights import AdaptedModel

foldseek = FoldSeekSetup(
    bin_dir="./bin",
    base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
).foldseek

dispatcher = InputDataDispatcher(
    DATASET_HOME="output/tasks/pair_regression/dataset",
    LMDB_HOME="output/tasks/pair_regression/lmdb",
    STRUCTURE_HOME="output/tasks/pair_regression/structures/",
    FOLDSEEK_PATH=foldseek,
    nproc=20,
)


def get_thermol_model():
    weight_worker = AdaptedModel(
        dir="./weights/SaProt/",
        huggingface_id="SaProtHub",
        model_name="Model-Structural_Similarity-650M",
        task_type="pair_regression",
        num_of_categories=10,
    )

    model, tokenizer = weight_worker.load_model()

    fitter = SADataAdapter(
        model=weight_worker,
        dataset_source="Multiple_pairs_of_PDB/CIF_Structures",
    )

    seqs: tuple[StructuralAwareSequencePair] = dispatcher.parse_data(
        "Multiple_pairs_of_PDB/CIF_Structures",
        "upload_files/[EXAMPLE]Multiple_pairs_of_PDB_Structures.csv",
    )

    outputs_list = []

    for i, s in track(enumerate(fitter(seqs))):
        input_1 = tokenizer(s[0], return_tensors="pt")
        input_1 = {k: v.to(model.device) for k, v in input_1.items()}
        input_2 = tokenizer(s[1], return_tensors="pt")
        input_2 = {k: v.to(model.device) for k, v in input_2.items()}

        with torch.no_grad():
            outputs = model(input_1, input_2)
        outputs_list.append(outputs)
    print(outputs_list)
    output_list = [output.squeeze().tolist() for output in outputs_list]
    for index, output in enumerate(output_list):
        print(f"For Sequence {index}, Prediction: Value {output}")


def main():
    get_thermol_model()


if __name__ == "__main__":
    main()

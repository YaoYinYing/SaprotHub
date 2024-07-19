import os

import torch
from saprot.utils.data_preprocess import InputDataDispatcher
from saprot.utils.foldseek_util import FoldSeekSetup, FoldSeek
from saprot.utils.weights import AdaptedModel
from saprot.utils.middleware import SADataAdapter

foldseek = FoldSeekSetup(
    bin_dir="./bin",
    base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
).foldseek

dispatcher = InputDataDispatcher(
    DATASET_HOME="output/tasks/regression/dataset",
    LMDB_HOME="output/tasks/regression/lmdb",
    STRUCTURE_HOME="output/tasks/regression/structures/",
    FOLDSEEK_PATH=foldseek,
    nproc=20,
)


def get_thermol_model():
    weight_worker = AdaptedModel(
        dir="./weights/SaProt/",
        huggingface_id="SaProtHub",
        model_name="Model-Thermostability-650M",
        task_type="regression",
        num_of_categories=10,
    )

    fitter = SADataAdapter(
        model=weight_worker, dataset_source="Multiple_SA_Sequences"
    )

    model, tokenizer = weight_worker.load_model()

    seqs = dispatcher.parse_data(
        "Multiple_SA_Sequences",
        "upload_files/[EXAMPLE]Multiple_SA_Sequences.csv",
    )

    outputs_list = []

    for i, s in enumerate(fitter(seqs)):
        inputs = tokenizer(s, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(inputs)
        outputs_list.append(outputs)

    outputs = [output.squeeze().tolist() for output in outputs_list]

    for index, output in enumerate(outputs_list):
        print(f"For Sequence {index}, Prediction: Value {output.item()}")


def main():
    get_thermol_model()


if __name__ == "__main__":
    main()

import os

import torch
from saprot.utils.foldseek_util import FoldSeekSetup,FoldSeek
from saprot.utils.weights import AdaptedModel

def get_thermol_model():
    weight_worker = AdaptedModel(
        dir="./weights/SaProt/",
        huggingface_id="SaProtHub",
        model_name="Model-Thermostability-650M",
        task_type='regression',
        num_of_categories=10
    )

    # weight_worker.device=torch.device('cpu')

    foldseek = FoldSeekSetup(
        bin_dir="./bin",
        base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
    ).foldseek

    pdb_dir='example/tmalign/inverse_folding_refold'

    model,tokenizer=weight_worker.load_model()

    pdbs=[i for i in os.listdir(pdb_dir) if i.endswith('.pdb')]

    seqs=[FoldSeek(foldseek, plddt_mask=False).query(os.path.join(pdb_dir,pdb))['A'] for pdb in pdbs]

    outputs_list=[]

    for i, s in enumerate(seqs):
        inputs = tokenizer(s.combined_sequence, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad(): 
            outputs = model(inputs)
        outputs_list.append(outputs)


    outputs= [output.squeeze().tolist() for output in outputs_list]

    for index, output in enumerate(outputs_list):
        print(f"For Sequence {index}, Prediction: Value {output.item()}")





def main():

    

    get_thermol_model()


if __name__ == "__main__":
    main()

import os

import torch
from saprot.utils.foldseek_util import FoldSeekSetup,FoldSeek
from saprot.utils.weights import AdaptedModel
from deep_mutagenese_scan import run_dms

def get_thermol_model():
    weight_worker = AdaptedModel(
        dir="/Users/yyy/.REvoDesign/weights/SaProt/",
        huggingface_id="SaProtHub",
        model_name="Model-Thermostability-650M",
        task_type='classification',
        num_of_categories=10
    )

    foldseek = FoldSeekSetup(
        bin_dir="./bin",
        base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
    ).foldseek

    model,tokenizer=weight_worker.load_model()

    pdbs=[i for i in os.listdir('example/tmalign/inverse_folding_refold') if i.endswith('.pdb')]

    seqs=[FoldSeek(foldseek, plddt_mask=False).query(pdb)['A'] for pdb in pdbs]

    outputs_list=[]

    for i, s in enumerate(seqs):
        inputs = tokenizer(s, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad(): 
            outputs = model(inputs)
        outputs_list.append(outputs)


    print(outputs_list)






def main():

    

    get_thermol_model()


if __name__ == "__main__":
    main()

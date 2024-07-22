import os

import numpy as np

from saprot.utils.downloader import StructureDownloader

from saprot.utils.dataclasses import UniProtID,UniProtIDs, Mask
from saprot.utils.foldseek_util import FoldSeek, FoldSeekSetup, PlddtMasker

uniprot_ids: tuple = (
        "O95905",
        "P04439",
        "P08246",
        "P42694",
        "Q6ZWK4",
        "Q6FHJ7",
        "P15529",
        "Q08648",
        "Q9Y2H6",
        "P63244",
    )

uniprot_ids_low_confidence: tuple[str]=(
    'A0A6G1QXL2',
    'Q9M1K1',
    'Q9M1K0',
    'Q9FIP9',
    'Q9C8Z9',
    'Q9LSE2',
    'Q9C707'
)
    

def get_af2_pdbs(uniprot_ids: tuple[str])-> str:
    

    uniprot_id_objs: tuple[UniProtID] = tuple(UniProtID(uniprot_id=_id, uniprot_type="AF2", chain_id='A') for _id in uniprot_ids)

    output_dir="output/plddt_parser"

    structure_downloader = StructureDownloader(
        data_type="pdb", save_dir=output_dir
    )
    files = structure_downloader.run(uniprot_id_objs)

    return files


def test_plddt_parser():

    files= get_af2_pdbs(uniprot_ids)
    
    plddt_parser = PlddtMasker(mask_cutoff=70.0)
    plddts_masks=tuple(plddt_parser.run(files))
    print(plddts_masks)



def test_plddt_masked_SA(uniprot_ids):

    files= get_af2_pdbs(uniprot_ids)

    foldseek = FoldSeekSetup(
        bin_dir="./bin",
        base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
    ).foldseek

    parsed_seqs = FoldSeek(foldseek,plddt_threshold=70).parallel_queries(pdb_files=files, enable_plddt_masks=True)

    print(parsed_seqs)



def main():
    test_plddt_parser()

    test_plddt_masked_SA(uniprot_ids)
    test_plddt_masked_SA(uniprot_ids_low_confidence)


if __name__ == "__main__":
    main()

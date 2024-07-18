from saprot.utils.tmalign import TMalignSetup, TMalign

import os
import pooch


def compile_tmalign():
    bin_path = "./bin/TMalign"

    compile_worker = TMalignSetup(bin_path)
    return compile_worker.ensure_binary()


def test_tmalign(tmalign_path: str):
    candidate_pdb_dir = "example/tmalign/inverse_folding_refold"

    template_pdb_url = "https://files.rcsb.org/download/1UBQ.pdb"
    template_pdb_path = pooch.retrieve(
        url=template_pdb_url,
        known_hash=None,
        fname=os.path.basename(template_pdb_url),
        path="example/tmalign",
    )

    candidate_pdbs = [
        f for f in os.listdir(candidate_pdb_dir) if f.endswith(".pdb")
    ]

    tmalign_worker = TMalign(tmalign_path)
    for pdb in candidate_pdbs:
        r = tmalign_worker.align(
            pdb1=template_pdb_path, pdb2=os.path.join(candidate_pdb_dir, pdb)
        )
        print(r)


def main():
    tm_align_binary = compile_tmalign()

    test_tmalign(tmalign_path=tm_align_binary)


if __name__ == "__main__":
    main()

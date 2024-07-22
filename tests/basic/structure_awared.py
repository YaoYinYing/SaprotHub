from saprot.utils.dataclasses import Mask
from saprot.utils.foldseek_util import FoldSeekSetup, FoldSeek


def main():
    pdb_path = "example/8ac8.cif"

    # Extract the "A" chain from the pdb file and encode it into a struc_seq
    # pLDDT is used to mask low-confidence regions if "plddt_mask" is True. Please set it to True when
    # use AF2 structures for best performance.

    foldseek = FoldSeekSetup(
        bin_dir="./bin",
        base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
    ).foldseek

    print(foldseek)
    parsed_seqs = FoldSeek(foldseek, ["A"]).query(pdb_file=pdb_path)

    parsed_seqs = parsed_seqs.get("A")

    print(f"{parsed_seqs=}")
    print(f"seq: {parsed_seqs.amino_acid_seq}")
    print(f"foldseek_seq: {parsed_seqs.structural_seq}")
    print(f"combined_seq: {parsed_seqs.combined_sequence}")

    mask = Mask(mask_pos_range="1-10,20-30,40-50")

    print(parsed_seqs.masked_seq(mask))
    print(parsed_seqs.masked_struct_seq(mask))

    reversed_mask_0 = mask.reversed()
    reversed_mask_1 = mask.reversed(
        full_length=len(parsed_seqs.amino_acid_seq)
    )
    reversed_mask_2 = mask.reversed(full_length=100)

    print(f"{mask=}")
    print(f"{reversed_mask_0=}")
    print(f"{reversed_mask_1=}")
    print(f"{reversed_mask_2=}")

    print(parsed_seqs.masked_seq(mask))
    print(parsed_seqs.masked_seq(reversed_mask_0))
    print(parsed_seqs.masked_seq(reversed_mask_1))
    print(parsed_seqs.masked_seq(reversed_mask_2))


if __name__ == "__main__":
    main()

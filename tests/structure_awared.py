from saprot.utils.foldseek_util import get_struc_seq,FoldSeekSetup, Mask, FoldSeek

def main():
    pdb_path = "example/8ac8.cif"

    # Extract the "A" chain from the pdb file and encode it into a struc_seq
    # pLDDT is used to mask low-confidence regions if "plddt_mask" is True. Please set it to True when
    # use AF2 structures for best performance.

    foldseek=FoldSeekSetup(bin_dir='./bin',base_url='https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/').foldseek

    print(foldseek)
    parsed_seqs = FoldSeek(foldseek, ["A"], plddt_mask=False).query(pdb_file=pdb_path)
    #print(parsed_seqs)

    parsed_seqs = parsed_seqs.get("A")

    print(f'{parsed_seqs=}')
    print(f"seq: {parsed_seqs.amino_acid_seq}")
    print(f"foldseek_seq: {parsed_seqs.structural_seq}")
    print(f"combined_seq: {parsed_seqs.combined_sequence}")

    mask=Mask(mask_pos_range='1-10,20-30,40-50')

    print(parsed_seqs.masked_seq(mask))
    print(parsed_seqs.masked_struct_seq(mask))

if __name__ == "__main__":
    main()
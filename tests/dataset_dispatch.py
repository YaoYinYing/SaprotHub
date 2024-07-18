import pdb
from saprot.utils.data_preprocess import InputDataDispatcher
from saprot.utils.foldseek_util import FoldSeekSetup

single_aa_seq = 'NSPRIVQSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYYESFPWFIKPAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCQYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITS'
single_sa_seq = 'NdSfPqRkIaVkQfSaNvDqLqTlElAkAaYwSeLaSaRpDlQlKvRlMvLvYsLvFqVlDsQcIlRaKvSvDvDqGqIkCdEkIdHaVlAvKvYsAcEvIvFlGvLhTdSsAvEvAsSvKvDsIvRvQvArLlKvSvFlApGpKtEwViVaFgYvYhEdSiFdPrWqFfIpKdPrAwHdSaPpSdRvGrLmYiSiVtHgIgNdPsYvLrIsPvFrFrIpGpLdQdNdRnFmTfQiFdRgLpSqEqTcKsEqIqTrNgPvYlAlMvRsLvYvErSvLqCsQsYaRaKdPpDqGqSwGgIkVdSkLdKwIlDvWsIsIcErRrYnQvLdPdQpSqYcQrRpMpPvDsFvRcRpRvFpLvQpVvCsVlNvErIsNcSvRrTgPqMkRnLkSdYwIdEwKdKdKdGpRpQdTtTtHiIiVmFmStFiRgDgIvTvSd'
uniprot_ids=["P76011", "Q5VSL9"]
pdb_ids=["1UBQ", "8AC8"]

foldseek = FoldSeekSetup(
    bin_dir="./bin",
    base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
).foldseek

dispatcher=InputDataDispatcher(
    DATASET_HOME="output/datadispatcher/dataset",
    LMDB_HOME="output/datadispatcher/lmdb",
    STRUCTURE_HOME="output/datadispatcher/structures",
    FOLDSEEK_PATH=foldseek,
    nproc=20,
)

def test_single_aa():
    print(dispatcher.parse_data('Single_AA_Sequence', single_aa_seq))

def test_single_sa():
    print(dispatcher.parse_data('Single_SA_Sequence', single_sa_seq))

def test_single_unprotid():
    print(dispatcher.parse_data("Single_UniProt_ID", uniprot_ids[0]))

def test_single_pdb_id():
    print(dispatcher.parse_data("Single_PDB/CIF_Structure", (pdb_ids[0], 'PDB','A')))

def test_multiple_pdb_ids():
    print(dispatcher.parse_data("Multiple_PDB/CIF_Structures", 'upload_files/[EXAMPLE]Multiple_PDB_Structures.csv'))

def test_multiple_uniprot_ids():
    print(dispatcher.parse_data("Multiple_UniProt_IDs", 'upload_files/[EXAMPLE]Multiple_UniProt_IDs.csv'))

def test_multiple_aa():
    print(dispatcher.parse_data("Multiple_AA_Sequences", 'upload_files/[EXAMPLE]Multiple_AA_Sequences.csv'))

def test_multiple_sa():
    print(dispatcher.parse_data("Multiple_SA_Sequences", 'upload_files/[EXAMPLE]Multiple_SA_Sequences.csv'))

def test_multiple_paired_AA():
    print(dispatcher.parse_data("Multiple_pairs_of_AA_Sequences", 'upload_files/[EXAMPLE]Multiple_pairs_of_AA_Sequences.csv'))

def test_multiple_paired_PDB():
    print(dispatcher.parse_data("Multiple_pairs_of_PDB/CIF_Structures", 'upload_files/[EXAMPLE]Multiple_pairs_of_PDB_Structures.csv'))

def main():
    test_multiple_uniprot_ids()
    test_multiple_paired_PDB()
    test_multiple_paired_AA()
    test_multiple_sa()
    test_multiple_aa()


    test_single_aa()
    test_single_sa()
    test_single_unprotid()
    test_single_pdb_id()
    test_multiple_pdb_ids()

if __name__ == "__main__":
    main()
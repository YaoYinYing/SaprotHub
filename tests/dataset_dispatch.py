import pdb
from saprot.utils.data_preprocess import InputDataDispatcher
from saprot.utils.foldseek_util import FoldSeekSetup

single_aa_seq = 'NSPRIVQSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYYESFPWFIKPAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCQYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITS'
single_sa_seq = 'NdSfPqRkIaVkQfSaNvDqLqTlElAkAaYwSeLaSaRpDlQlKvRlMvLvYsLvFqVlDsQcIlRaKvSvDvDqGqIkCdEkIdHaVlAvKvYsAcEvIvFlGvLhTdSsAvEvAsSvKvDsIvRvQvArLlKvSvFlApGpKtEwViVaFgYvYhEdSiFdPrWqFfIpKdPrAwHdSaPpSdRvGrLmYiSiVtHgIgNdPsYvLrIsPvFrFrIpGpLdQdNdRnFmTfQiFdRgLpSqEqTcKsEqIqTrNgPvYlAlMvRsLvYvErSvLqCsQsYaRaKdPpDqGqSwGgIkVdSkLdKwIlDvWsIsIcErRrYnQvLdPdQpSqYcQrRpMpPvDsFvRcRpRvFpLvQpVvCsVlNvErIsNcSvRrTgPqMkRnLkSdYwIdEwKdKdKdGpRpQdTtTtHiIiVmFmStFiRgDgIvTvSd'

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


def main():
    test_single_aa()
    test_single_sa()

if __name__ == "__main__":
    main()
import os
from pathlib import Path
from typing import Any, List, Literal, Tuple, Union
import warnings
import pandas as pd
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from string import ascii_uppercase, ascii_lowercase

from immutabledict import immutabledict
from dataclasses import dataclass

from saprot.utils.foldseek_util import (
    StructuralAwareSequence,
    get_struc_seq,
    StructuralAwareSequences,
    Mask,
    FoldSeek,
)

from saprot.utils.mpr import MultipleProcessRunnerSimplifier

from huggingface_hub import snapshot_download


DATA_TYPES: tuple = (
    "Single_AA_Sequence",
    "Single_SA_Sequence",
    "Single_UniProt_ID",
    "Single_PDB/CIF_Structure",
    "Multiple_AA_Sequences",
    "Multiple_SA_Sequences",
    "Multiple_UniProt_IDs",
    "Multiple_PDB/CIF_Structures",
    "SaprotHub_Dataset",
    "A_pair_of_AA_Sequences",
    "A_pair_of_SA_Sequences",
    "A_pair_of_UniProt_IDs",
    "A_pair_of_PDB/CIF_Structures",
    "Multiple_pairs_of_AA_Sequences",
    "Multiple_pairs_of_SA_Sequences",
    "Multiple_pairs_of_UniProt_IDs",
    "Multiple_pairs_of_PDB/CIF_Structures",
)

DATA_TYPES_HINT = Literal[
    "Single_AA_Sequence",
    "Single_SA_Sequence",
    "Single_UniProt_ID",
    "Single_PDB/CIF_Structure",
    "Multiple_AA_Sequences",
    "Multiple_SA_Sequences",
    "Multiple_UniProt_IDs",
    "Multiple_PDB/CIF_Structures",
    "SaprotHub_Dataset",
    "A_pair_of_AA_Sequences",
    "A_pair_of_SA_Sequences",
    "A_pair_of_UniProt_IDs",
    "A_pair_of_PDB/CIF_Structures",
    "Multiple_pairs_of_AA_Sequences",
    "Multiple_pairs_of_SA_Sequences",
    "Multiple_pairs_of_UniProt_IDs",
    "Multiple_pairs_of_PDB/CIF_Structures",
]


@dataclass
class UniProtID:
    uniprot_id: str
    uniprot_type: Literal["AF2", "PDB"]
    chain_id: str

    SA_seq: StructuralAwareSequences = None

    @property
    def __post_init__(self):
        if self.uniprot_type not in ["AF2", "PDB"]:
            raise ValueError(
                f"UniProt type must be either 'AF2' or 'PDB', got {self.uniprot_type}"
            )

    @property
    def SA(self) -> StructuralAwareSequence:
        return self.SA_seq[self.chain_id]

    def set_sa_name(self, name):
        for sa in self.SA_seq.seqs.values():
            sa.name = name


@dataclass
class StructuralAwareSequencePair:
    seq_1: StructuralAwareSequence
    seq_2: StructuralAwareSequence

    @property
    def paired_sa(self):
        raise NotImplementedError


@dataclass
class UniProtIDs:
    uniprot_ids: Union[Tuple[UniProtID]]

    def __post_init__(self):
        if isinstance(self.uniprot_ids, UniProtID):
            self.uniprot_ids = [self.uniprot_ids]

    @property
    def all_labels(self) -> Tuple[str]:
        return tuple(uniprot.uniprot_id for uniprot in self.uniprot_ids)

    @property
    def is_AF2_structures(self) -> bool:
        return all(x.uniprot_type == "AF2" for x in self.uniprot_ids)

    def map_sa_to_uniprot_ids(self, sa_seqs: Tuple[StructuralAwareSequences]):

        if not len(sa_seqs) == len(self.uniprot_ids):
            raise ValueError(
                f"The number of uniprot ids ({len(self.uniprot_ids)}) and the number of sa_seqs ({len(sa_seqs)}) must be equal."
            )

        for i, sa in enumerate(sa_seqs):
            self.uniprot_ids[i].SA_seq = sa
        return

    @property
    def SA_seqs_as_tuple(self) -> Tuple[StructuralAwareSequence]:
        return tuple(x.SA for x in self.uniprot_ids)


@dataclass
class InputDataDispatcher:
    data_type: DATA_TYPES_HINT
    raw_data: Any
    DATASET_HOME: str
    LMDB_HOME: str
    STRUCTURE_HOME: str
    FOLDSEEK_PATH: str
    nproc: int = os.cpu_count()

    def UniProtID2SA(
        self, proteins: Union[List[UniProtID], Tuple[UniProtID], UniProtID]
    ) -> UniProtIDs:
        protein_list = UniProtIDs(proteins)
        files = self.uniprot2pdb(protein_list.uniprot_ids)

        foldseeq_runner = FoldSeek(
            self.FOLDSEEK_PATH,
            plddt_mask=protein_list.is_AF2_structures,
            nproc=self.nproc,
        )
        sas = foldseeq_runner.parallel_queries(files)
        protein_list.map_sa_to_uniprot_ids(sas)

        return protein_list

    def parse_data(
        self, data_type: DATA_TYPES_HINT, raw_data: Union[str, Tuple, List]
    ) -> Union[Tuple[StructuralAwareSequence], Tuple[StructuralAwareSequencePair]]:

        # 0. Single AA Sequence
        if data_type == "Single_AA_Sequence":
            input_seq: str = raw_data

            aa_seq = input_seq

            return (
                StructuralAwareSequence(
                    amino_acid_seq=aa_seq, structural_seq="#" * len(aa_seq)
                ),
            )

        # 1. Single SA Sequence
        if data_type == "Single_SA_Sequence":
            input_seq = raw_data
            sa_seq = input_seq

            return (StructuralAwareSequence(None, None).from_SA_sequence(sa_seq),)

        # 2. Single UniProt ID
        if data_type == "Single_UniProt_ID":
            input_seq = raw_data
            uniprot_id = input_seq

            protein = UniProtID(uniprot_id, "AF2", "A")
            protein_list = self.UniProtID2SA(proteins=protein)

            return protein_list.SA_seqs_as_tuple

        # 3. Single PDB/CIF Structure
        if data_type == "Single_PDB/CIF_Structure":
            uniprot_id, struc_type, chain = raw_data[:3]

            protein = UniProtID(uniprot_id, struc_type, chain)
            protein_list = self.UniProtID2SA(protein)

            return protein_list.SA_seqs_as_tuple

        # Multiple sequences
        # raw_data = upload_files/xxx.csv
        if data_type.startswith("Multiple"):
            uploaded_csv_path = raw_data
            csv_dataset_path = self.DATASET_HOME / uploaded_csv_path.name

        # 4. Multiple AA Sequences
        if data_type == "Multiple_AA_Sequences":
            protein_df = pd.read_csv(uploaded_csv_path)

            SA: List[StructuralAwareSequence] = []

            for index, aa_seq in protein_df["Sequence"].items():
                SA.append(
                    StructuralAwareSequence(
                        amino_acid_seq=aa_seq, structural_seq="#" * len(aa_seq)
                    )
                )

            return tuple(SA)

        # 5. Multiple SA Sequences
        if data_type == "Multiple_SA_Sequences":
            protein_df = pd.read_csv(uploaded_csv_path)

            SA: List[StructuralAwareSequence] = []

            for index, sa_seq in protein_df["Sequence"].items():
                SA.append(StructuralAwareSequence(None, None).from_SA_sequence(sa_seq))

            return tuple(SA)

        # 6. Multiple UniProt IDs
        if data_type == "Multiple_UniProt_IDs":
            protein_df = pd.read_csv(uploaded_csv_path)

            protein_ids = protein_df.iloc[:, 0].tolist()
            protein_list = self.UniProtID2SA(
                [UniProtID(_id, "AF2", "A") for _id in protein_ids]
            )

            return protein_list.SA_seqs_as_tuple

        # 7. Multiple PDB/CIF Structures
        if data_type == "Multiple_PDB/CIF_Structures":
            protein_df = pd.read_csv(uploaded_csv_path)

            id_col = protein_df["Sequence"].to_list()
            type_col = protein_df["type"].to_list()
            chain_col = protein_df["chain"].to_list()

            protein_list = self.UniProtID2SA(
                [
                    UniProtID(_id, _type, _chain)
                    for _id, _type, _chain in zip(id_col, type_col, chain_col)
                ]
            )

            return protein_list.SA_seqs_as_tuple

        # 8. SaprotHub Dataset
        elif data_type == "SaprotHub_Dataset":
            input_repo_id = raw_data
            REPO_ID = input_repo_id

            if REPO_ID.startswith("/"):
                return Path(REPO_ID)

            snapshot_download(
                repo_id=REPO_ID, repo_type="dataset", local_dir=self.LMDB_HOME / REPO_ID
            )

            dataset_dir=self.LMDB_HOME / REPO_ID
            csv_files=[x for x in os.listdir(dataset_dir) if x.endswith(".csv")]

            for csv in csv_files:
                raise NotImplementedError

            return 

        # 9. Pair Single AA Sequences
        elif data_type == "A_pair_of_AA_Sequences":

            return StructuralAwareSequencePair(
                *[
                    StructuralAwareSequence(
                        amino_acid_seq=aa_seq, structural_seq="#" * len(aa_seq)
                    )
                    for aa_seq in raw_data
                ]
            )

        # 10. Pair Single SA Sequences
        elif data_type == "A_pair_of_SA_Sequences":

            return StructuralAwareSequencePair(
                *[
                    StructuralAwareSequence(None, None).from_SA_sequence(sa_seq)
                    for sa_seq in raw_data
                ]
            )

        # 11. Pair Single UniProt IDs
        elif data_type == "A_pair_of_UniProt_IDs":

            protein_list = self.UniProtID2SA(
                proteins=[UniProtID(uniprot_id, "AF2", "A") for uniprot_id in raw_data]
            )
            return StructuralAwareSequencePair(*protein_list.SA_seqs_as_tuple)

        # 12. Pair Single PDB/CIF Structure
        if data_type == "A_pair_of_PDB/CIF_Structures":

            if raw_datalen := len(raw_data) % 3 != 0:
                raise ValueError(
                    "The length of the raw_data should be a multiple of 3."
                )

            uniprot_id_slice = slice(0, raw_datalen, 3)
            struc_type_slice = slice(1, raw_datalen, 3)
            chain_slice = slice(2, raw_datalen, 3)

            protein_list = self.UniProtID2SA(
                proteins=[
                    UniProtID(_id, _type, _chain)
                    for _id, _type, _chain in zip(
                        raw_data[uniprot_id_slice],
                        raw_data[struc_type_slice],
                        raw_data[chain_slice],
                    )
                ]
            )
            return StructuralAwareSequencePair(*protein_list.SA_seqs_as_tuple)

        # # Pair raw_data = upload_files/xxx.csv
        # if data_type in data_type_list[12:16]:
        #   uploaded_csv_path = raw_data
        #   csv_dataset_path = DATASET_HOME / uploaded_csv_path.name

        # 13. Pair Multiple AA Sequences
        if data_type == "Multiple_pairs_of_AA_Sequences":
            protein_df = pd.read_csv(uploaded_csv_path)

            pairs_1: list[StructuralAwareSequence] = [
                StructuralAwareSequence(
                    amino_acid_seq=aa_seq,
                    structural_seq="#" * len(aa_seq),
                    name="name_1",
                    chain="A",
                    skip_post_processing=True,
                )
                for aa_seq in protein_df["seq_1"].to_list()
            ]

            pairs_2: list[StructuralAwareSequence] = [
                StructuralAwareSequence(
                    amino_acid_seq=aa_seq,
                    structural_seq="#" * len(aa_seq),
                    name="name_2",
                    chain="A",
                    skip_post_processing=True,
                )
                for aa_seq in protein_df["seq_2"].to_list()
            ]

            return tuple(
                StructuralAwareSequencePair(p1, p2) for p1, p2 in zip(pairs_1, pairs_2)
            )

        # 14. Pair Multiple SA Sequences
        if data_type == "Multiple_pairs_of_SA_Sequences":
            protein_df = pd.read_csv(uploaded_csv_path)

            pairs_1: list[StructuralAwareSequence] = [
                StructuralAwareSequence(
                    amino_acid_seq=None,
                    structural_seq=None,
                    name="name_1",
                    chain="A",
                    skip_post_processing=True,
                ).from_SA_sequence(sa_seq)
                for sa_seq in protein_df["seq_1"].to_list()
            ]

            pairs_2: list[StructuralAwareSequence] = [
                StructuralAwareSequence(
                    amino_acid_seq=None,
                    structural_seq=None,
                    name="name_2",
                    chain="A",
                    skip_post_processing=True,
                ).from_SA_sequence(sa_seq)
                for sa_seq in protein_df["seq_2"].to_list()
            ]

            return tuple(
                StructuralAwareSequencePair(p1, p2) for p1, p2 in zip(pairs_1, pairs_2)
            )

        # 15. Pair Multiple UniProt IDs
        if data_type == "Multiple_pairs_of_UniProt_IDs":

            protein_df = pd.read_csv(uploaded_csv_path)

            protein_ids = protein_df.loc[:, "seq_1"].tolist()
            protein_list_1 = self.UniProtID2SA(
                [UniProtID(_id, "AF2", "A") for _id in protein_ids]
            )
            for x in protein_list_1.uniprot_ids:
                x.set_sa_name("name_1")

            protein_ids = protein_df.loc[:, "seq_2"].tolist()
            protein_list_2 = self.UniProtID2SA(
                [UniProtID(_id, "AF2", "A") for _id in protein_ids]
            )
            for x in protein_list_2.uniprot_ids:
                x.set_sa_name("name_2")

            return tuple(
                StructuralAwareSequencePair(p1, p2)
                for p1, p2 in zip(
                    protein_list_1.SA_seqs_as_tuple, protein_list_2.SA_seqs_as_tuple
                )
            )

        # # 13-16. Pair Multiple Sequences
        # elif data_type in data_type_list[12:16]:
        #   print(Fore.BLUE+f"Please upload the .csv file which contains {data_type}"+Style.RESET_ALL)
        #   uploaded_csv_path = upload_file(UPLOAD_FILE_HOME)
        #   print(Fore.BLUE+"Successfully upload your .csv file!"+Style.RESET_ALL)
        #   print("="*100)

        elif data_type == "Multiple_pairs_of_PDB/CIF_Structures":
            protein_df = pd.read_csv(uploaded_csv_path)
            # columns: seq_1, seq_2, type_1, type_2, chain_1, chain_2, label, stage

            protein_pair: list[UniProtIDs] = []
            for i in range(1, 3):
                protein_table: list[UniProtID] = []
                for index, row in protein_df.iterrows():
                    row_tuple = (row[f"seq_{i}"], row[f"type_{i}"], row[f"chain_{i}"])
                    protein_table.append(UniProtID(*row_tuple))

                p: UniProtIDs = self.UniProtID2SA(protein_table)
                for x in p.uniprot_ids:
                    x.set_sa_name(f"name_{i}")

                protein_pair.append(p)

            return tuple(
                StructuralAwareSequencePair(s1, s2)
                for s1, s2 in zip(
                    protein_pair[0].SA_seqs_as_tuple, protein_pair[1].SA_seqs_as_tuple
                )
            )

    ################################################################################
    ########################## Download predicted structures #######################
    ################################################################################
    def uniprot2pdb(self, uniprot_ids, nprocess=20) -> Tuple[str]:
        from saprot.utils.downloader import AlphaDBDownloader

        os.makedirs(self.STRUCTURE_HOME, exist_ok=True)
        # check exists files
        exists_file = set(
            [
                x
                for x in os.listdir(self.STRUCTURE_HOME)
                if x.endswith(".pdb") or x.endswith(".cif")
            ]
        )
        if exists_file:
            warnings.warn(
                f"{len(exists_file)} files already exists in {self.STRUCTURE_HOME}"
            )

        af2_downloader = AlphaDBDownloader(
            uniprot_ids, "pdb", save_dir=self.STRUCTURE_HOME, n_process=nprocess
        )
        af2_downloader.run()

        updated_files = set(
            [
                x
                for x in os.listdir(self.STRUCTURE_HOME)
                if x.endswith(".pdb") or x.endswith(".cif")
            ]
        )

        return Tuple(updated_files.difference(exists_file))


alphabet_list = list(ascii_uppercase + ascii_lowercase)


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    outputs["plddt"] *= 100

    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

import contextlib
from dataclasses import dataclass, field
import os
import platform
import shutil
import subprocess
import tarfile
from typing import Any, Literal, Optional, Union
import numpy as np
import re
import pooch
import platformdirs
import tempfile
import subprocess
from joblib import Parallel, delayed
from rich.progress import track

from saprot.utils.mask_tool import shorter_range


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


FOLDSEEK_STRUC_VOCAB = "pynwrqhgdlvtmfsaeikc#"


@dataclass
class Mask:
    """
    A class to represent a mask, used to mark specific positions in a sequence.

    Attributes:
        mask_pos_range: The range of positions marked by the mask, in string format.
        mask_label: The label used to mark positions, default is '#'.
        mask_separator: The separator used in the mask position range, default is ','.
        mask_connector: The connector used in the mask position range, default is '-'.
        zero_indexed: Indicates whether the position is 0-based, default is False.
    """

    mask_pos_range: str
    mask_label: str = "#"
    mask_separator: str = ","
    mask_connector: str = "-"
    zero_indexed: bool = False

    def from_masked(self, masked_sequence: str):
        """
        Updates the mask position range based on a masked sequence.

        Parameters:
            masked_sequence: A string containing the masked sequence.
        """
        mask: tuple[int] = tuple(
            i for i, k in enumerate(masked_sequence) if k == self.mask_label
        )
        self.mask_pos_range = shorter_range(
            mask, connector=self.mask_connector, seperator=self.mask_separator
        )
        print(f"Generate mask from masked sequence: {self.mask_pos_range}")
        self.zero_indexed = True

    def __post_init__(self):
        """
        Validates the mask_label after object initialization to ensure it is a single character.
        """
        if len(self.mask_label) != 1:
            raise ValueError("mask_label must be a single character")

    @property
    def mask(self) -> list[int]:
        """
        Gets the list of positions marked by the mask.

        Returns:
            A list of masked positions.
        Raises:
            ValueError: If mask_pos_range is empty.
        """
        if self.mask_pos_range == "":
            raise ValueError("Mask pos range cannot be empty!")
        from saprot.utils.mask_tool import expand_range

        expanded_mask_pos = expand_range(
            shortened_str=self.mask_pos_range,
            connector=self.mask_connector,
            seperator=self.mask_separator,
        )
        if not self.zero_indexed:
            expanded_mask_pos = [i - 1 for i in expanded_mask_pos]
        return expanded_mask_pos

    def masked(self, sequence: Union[str, list, tuple]) -> str:
        """
        Returns a masked sequence based on the current mask settings.

        Parameters:
            sequence: The original sequence, can be a string, list, or tuple.

        Returns:
            The masked sequence.
        Raises:
            ValueError: If the length of the sequence is less than the maximum position of the mask, or if the mask position range is empty.
        """
        sequence = list(sequence)
        if len(sequence) < max(self.mask) + 1:
            raise ValueError(
                f"Sequence length {len(sequence)} is less than the mask position range {self.mask_pos_range}"
            )
        if self.mask_pos_range is None or self.mask_pos_range == "":
            self.mask_pos_range = f"1-{len(sequence)}"
            print(f"Mask is set to full length: {self.mask_pos_range}")
        for pos in self.mask:
            sequence[pos] = self.mask_label
        return "".join(sequence)


@dataclass
class StructuralAwareSequence:
    """
    A class to represent a protein sequence that is aware of its structural context.

    Attributes:
    amino_acid_seq (str): The amino acid sequence of the protein.
    structural_seq (str): The structural sequence corresponding to the protein sequence.
    desc (str): A description of the sequence.
    name (str): The name of the sequence.
    name_chain (Optional[str]): The name of the chain within the sequence. Defaults to None.
    chain (Optional[str]): The identifier of the chain. Defaults to None.
    """

    amino_acid_seq: str
    structural_seq: str
    desc: Optional[str] = None
    name: Optional[str] = None
    name_chain: Optional[str] = None
    chain: Optional[str] = None

    skip_post_processing: Optional[bool] = False

    def __post_init__(self):
        """
        Post-initialization method to clean up sequences and extract chain information from the description.
        Validates that the amino acid sequence and structural sequence have the same length.
        Removes file extensions from the name if present.
        """

        if self.skip_post_processing:
            return

        if self.amino_acid_seq is None and self.structural_seq is None:
            return

        self.amino_acid_seq = self.amino_acid_seq.strip()
        self.structural_seq = self.structural_seq.strip().lower()
        self.name_chain = self.desc.split(" ")[0]
        self.chain = self.name_chain.replace(self.name, "").split("_")[-1]

        if len(self.amino_acid_seq) != len(self.structural_seq):
            raise ValueError(
                "The amino acid sequence and structural sequence must be of the same length"
            )

        if self.name.endswith(".cif") or self.name.endswith(".pdb"):
            self.name = self.name[:-4]
    @property
    def _blind(self):
        return self.structural_seq.strip('#') == '' 
    
    
    def from_SA_sequence(self, SA_sequence: str):
        seq_len = len(SA_sequence)
        if not (seq_len > 0 and seq_len % 2 == 0):
            raise ValueError("The SA sequence must be a multiple of 2")

        aa_seq_islice = slice(0, seq_len, 2)
        st_seq_islice = slice(1, seq_len, 2)

        self.amino_acid_seq = SA_sequence[aa_seq_islice]
        self.structural_seq = SA_sequence[st_seq_islice]

        return self
    

    @property
    def combined_sequence(self) -> str:
        """
        Generates a combined sequence string where each amino acid is followed by its corresponding structural letter.

        Returns:
        str: The combined sequence of amino acids and structures.
        """
        combined_sequence = "".join(
            f"{_seq}{_struc_seq}"
            for _seq, _struc_seq in zip(
                self.amino_acid_seq, self.structural_seq.lower()
            )
        )
        return combined_sequence
    

    def masked_seq(self, mask: "Mask") -> str:
        """
        Masks the amino acid sequence based on the provided mask object.

        Parameters:
        mask (Mask): An instance of the Mask class that determines which parts of the sequence should be masked.

        Returns:
        str: The masked amino acid sequence.
        """
        return mask.masked(self.amino_acid_seq)

    def masked_struct_seq(self, mask: "Mask") -> str:
        """
        Masks the structural sequence based on the provided mask object.

        Parameters:
        mask (Mask): An instance of the Mask class that determines which parts of the sequence should be masked.

        Returns:
        str: The masked structural sequence.
        """
        return mask.masked(self.structural_seq)


@dataclass
class StructuralAwareSequences:
    """
    A class for managing sequences that are aware of their structure.

    This class stores a source structure identifier and a mapping of chains to sequences,
    allowing for filtering and chain-based sequence access.

    Attributes:
    source_structure (str): Identifier for the source structure, could be a filename or a unique identifier for the structure.
    seqs (dict[str, StructuralAwareSequence]): A dictionary mapping chain identifiers to StructuralAwareSequence objects representing sequences of specific chains.

    """

    source_structure: str = None
    seqs: dict[str, StructuralAwareSequence] = field(default_factory=dict)

    foldseek_results: tuple[str] = None

    def __str__(self):
        return f"Source structure: {self.source_structure}\nSequences: {self.seqs}"

    def filtered(self, chains: tuple[str]) -> "StructuralAwareSequences":
        """
        Returns a filtered view of StructuralAwareSequences containing only the specified chains.

        Parameters:
        chains: A tuple of strings containing the identifiers of the chains to retain.
        """
        return StructuralAwareSequences(
            source_structure=self.source_structure,
            seqs={chain: seq for chain, seq in self.seqs.items() if chain in chains},
        )

    def __getitem__(self, chain_id: str):
        """
        Enables accessing sequences by chain identifier using indexing syntax.

        Parameter:
        chain_id: The identifier of the chain to access.

        Returns:
        The StructuralAwareSequence associated with the given chain identifier.
        """
        return self.seqs[chain_id]

    def get(self, chain_id: str, default_value: Union[Any, None] = None):
        """
        Retrieves a sequence by chain identifier with a fallback default value.

        Parameters:
        chain_id: The identifier of the chain to retrieve.
        default_value: The value to return if the chain is not found (default is None).

        Returns:
        The StructuralAwareSequence for the given chain identifier or the default value if not found.
        """
        if chain_id not in self.seqs:
            return default_value
        return self.seqs[chain_id]


@dataclass
class FoldSeekSetup:
    """
    A configuration class for the FoldSeek tool, used to determine the
    download and setup of the FoldSeek binary based on the operating system
    and CPU architecture.

    Attributes:
        bin_dir (str): Directory where the FoldSeek binary will be stored.
        bin_dir: The directory where the FoldSeek binary is located.
        base_url (str, optional): Base URL from which to download FoldSeek binaries. Defaults to 'https://mmseqs.com/foldseek/'.
        os_build (str, optional): Operating system identifier. Defaults to 'linux'.
        arch (str): The machine's architecture as reported by the platform.uname() function.
        bin_path (str): Path to the FoldSeek binary, constructed by joining bin_dir and 'foldseek'.
    """

    bin_dir: str
    base_url: str = "https://mmseqs.com/foldseek/"

    # Gather system information
    uname = platform.uname()
    cpu_info = subprocess.run(
        ["cat", "/proc/cpuinfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.decode("utf8")

    arch: str = uname.machine

    os_build: str = uname.system

    def __post_init__(self):
        """
        Post-initialization method that sets up the FoldSeek binary.
        """

        if not os.path.exists(self.bin_dir):
            os.makedirs(self.bin_dir)

    @property
    def bin_path(self) -> str:
        return os.path.join(self.bin_dir, "foldseek")

    def build_with_flag(self, flag: Literal["avx2", "sse2"]) -> bool:
        """
        Checks if the CPU supports a given instruction set.

        Args:
            flag (Literal['avx2', 'sse2']): Instruction set to check support for.

        Returns:
            bool: True if the CPU supports the instruction set, False otherwise.
        """
        return flag in self.cpu_info

    @property
    def retrieve_url(self) -> str:
        """
        Determines the download URL for the FoldSeek binary based on the OS and CPU architecture.

        Returns:
            str: The URL to download the appropriate FoldSeek binary.

        Raises:
            NotImplementedError: If the platform or architecture is unsupported.
        """
        if self.os_build == "Darwin":
            return f"{self.base_url}/foldseek-osx-universal.tar.gz"
        if self.os_build == "Linux":
            if self.arch == "x86_64":
                if self.build_with_flag("sse2"):
                    return f"{self.base_url}/foldseek-linux-sse2.tar.gz"
                if self.build_with_flag("avx2"):
                    return f"{self.base_url}/foldseek-linux-avx2.tar.gz"
            if self.arch == "aarch64":
                return f"{self.base_url}/foldseek-linux-arm64.tar.gz"
        raise NotImplementedError(
            f"Unsupported platform {self.os_build} or architecture {self.arch}"
        )

    @property
    def foldseek(self) -> str:
        """
        Ensures the FoldSeek binary exists at the specified path.

        Returns:
            str: Path to the FoldSeek binary.

        Raises:
            RuntimeError: If the binary cannot be retrieved.
        """
        if not os.path.exists(self.bin_path):
            self.get_foldseek_binary()
        return self.bin_path

    def get_foldseek_binary(self) -> str:
        """
        Downloads and extracts the FoldSeek binary to the specified directory.

        Returns:
            str: Path to the downloaded and extracted binary.

        Raises:
            RuntimeError: If the binary cannot be retrieved after extraction.
        """
        compressed_file_path = pooch.retrieve(
            self.retrieve_url, known_hash=None, progressbar=True
        )
        with tarfile.open(compressed_file_path, "r:gz") as tar:
            p = platformdirs.user_cache_dir("FoldSeek")
            tar.extractall(path=p)

            os.rename(os.path.join(p, "foldseek", "bin", "foldseek"), self.bin_path)

        os.remove(compressed_file_path)
        if not os.path.exists(self.bin_path):
            raise RuntimeError("Could not retrieve foldseek binary")

        # chmod +x
        os.chmod(self.bin_path, 0o555)


@dataclass
class FoldSeek:
    foldseek: str
    nproc: int = os.cpu_count()
    plddt_mask: bool = False
    plddt_threshold: float = 70.0
    name: str = None

    def __post_init__(self):
        if not os.path.exists(self.foldseek):
            raise FileNotFoundError(f"Foldseek not found: {self.foldseek}")

    def query(
        self, pdb_file: str, chain_ids: tuple[str] = None
    ) -> StructuralAwareSequences:
        if pdb_file is None or not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        self.name = os.path.basename(pdb_file)
        if self.plddt_mask:
            plddts = extract_plddt(pdb_file)
            # Mask regions with plddt < threshold
            indices = np.where(plddts < self.plddt_threshold)[0]

        with tmpdir_manager() as tmpdir:
            tmp_save_path = os.path.join(tmpdir, "get_struc_seq.tsv")
            cmd = [
                self.foldseek,
                "structureto3didescriptor",
                "-v",
                "0",
                "--threads",
                "1",
                "--chain-name-mode",
                "1",
                pdb_file,
                tmp_save_path,
            ]

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            stdout, stderr = process.communicate()
            retcode = process.wait()

            if retcode and not (result_exists := os.path.exists(tmp_save_path)):
                print(f"FoldSeek failed. \nFull Command:\n{cmd}\n  stderr begin:")
                for error_line in stderr.decode("utf-8").splitlines():
                    if error_line.strip():
                        print(error_line.strip())
                print("stderr end")

                raise RuntimeError(
                    f"FoldSeek failed. \n"
                    f"Full Command:\n{cmd}\n"
                    f"return code: {retcode}\n"
                    f"Results file: {result_exists}\n\n"
                    f"stdout:\n{stdout.decode('utf-8')}\n\n"
                    f"stderr:\n{stderr[:500_000].decode('utf-8')}\n"
                )

            ret = tuple(open(tmp_save_path, "r").read().strip().split("\n"))

        seq_dict = StructuralAwareSequences(
            source_structure=pdb_file,
            seqs={},
            foldseek_results=ret,
        )

        for i, line in enumerate(seq_dict.foldseek_results):
            desc, seq, struc_seq = line.split("\t")[:3]

            new_seq = StructuralAwareSequence(
                amino_acid_seq=seq,
                structural_seq=struc_seq,
                desc=desc,
                name=self.name,
            )

            # Mask low plddt
            if self.plddt_mask:
                np_seq = np.array(list(new_seq.structural_seq))
                np_seq[indices] = "#"
                new_seq.structural_seq = "".join(np_seq)

            seq_dict.seqs.update({new_seq.chain: new_seq})

        return (
            seq_dict
            if chain_ids is None or len(chain_ids) == 0
            else seq_dict.filtered(chains=tuple(chain_ids))
        )

    def parallel_queries(
        self, pdb_files: Union[tuple, list]
    ) -> tuple[StructuralAwareSequences]:
        return tuple(
            Parallel(n_jobs=self.nproc)(
                delayed(self.query)(pdb_file)
                for pdb_file in track(pdb_files, description="FoldSeeking...")
            )
        )


# TODO: multple chain?
def extract_plddt(pdb_path: str) -> np.ndarray:
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """
    with open(pdb_path, "r") as r:
        plddt_dict = {}
        for line in r:
            line = re.sub(" +", " ", line).strip()
            splits = line.split(" ")

            if splits[0] == "ATOM":
                # If position < 1000
                if len(splits[4]) == 1:
                    pos = int(splits[5])

                # If position >= 1000, the blank will be removed, e.g. "A 999" -> "A1000"
                # So the length of splits[4] is not 1
                else:
                    pos = int(splits[4][1:])

                plddt = float(splits[-2])

                if pos not in plddt_dict:
                    plddt_dict[pos] = [plddt]
                else:
                    plddt_dict[pos].append(plddt)

    plddts = np.array([np.mean(v) for v in plddt_dict.values()])
    return plddts

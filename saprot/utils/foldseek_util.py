import contextlib
from dataclasses import dataclass
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

import warnings
from joblib import Parallel, delayed
from rich.progress import track

from saprot.utils.dataclasses import (
    Mask,
    StructuralAwareSequence,
    StructuralAwareSequences,
)

from Bio.PDB import PDBParser
import numpy as np

from saprot.utils.mask_tool import shorter_range


@dataclass
class PlddtMasker:

    nproc: int = os.cpu_count()
    mask_cutoff: float = (
        0.0  # residue id that has plddt lower than this value will be masked.
    )

    def __post_init__(self):
        if self.mask_cutoff > 100:
            warnings.warn("Mask cutoff should be between 0 and 100")
            self.mask_cutoff = 100

        if self.mask_cutoff < 0:
            warnings.warn("Mask cutoff should be between 0 and 100")
            self.mask_cutoff = 0

    def parse_pdb(self, pdb_file: str) -> dict[str, Mask]:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("model", pdb_file)
        chain_plddt = {}

        for model in structure:
            for chain in model:
                chain_id = chain.id
                plddt_values = []

                plddt_values = [
                    residue["CA"].get_bfactor() for residue in chain if "CA" in residue
                ]

                # a pre mask for non residues
                mask = Mask(mask_pos_range=None)
                if self.mask_cutoff >= 0 and self.mask_cutoff <= 100:
                    plddt_values_masked_slice = [
                        i for i, v in enumerate(plddt_values) if v < self.mask_cutoff
                    ]
                    #print(plddt_values_masked_slice)
                    # update mask if it is a valid slice
                    if plddt_values_masked_slice != []:
                        mask = Mask(
                            mask_pos_range=shorter_range(
                                list(plddt_values_masked_slice)
                            ),
                            zero_indexed=True,
                        )

                chain_plddt[chain_id] = mask

        return chain_plddt

    def run(self, payload: tuple) -> tuple[dict[str, Mask]]:
        return Parallel(n_jobs=self.nproc)(
            delayed(self.parse_pdb)(pdb_file) for pdb_file in track(payload)
        )


class PlddtMaskWarning(Warning): ...


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
        ["cat", "/proc/cpuinfo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
    plddt_threshold: float = 70.0
    name: str = None

    drop_fold_results_after_initialized: bool = True

    def __post_init__(self):
        if not os.path.exists(self.foldseek):
            raise FileNotFoundError(f"Foldseek not found: {self.foldseek}")

    def query(
        self, pdb_file: str, chain_ids: tuple[str] = None, enable_plddt_mask: bool = False
    ) -> StructuralAwareSequences:
        if pdb_file is None or not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        self.name = os.path.basename(pdb_file)
        if enable_plddt_mask:
            plddt_masks = PlddtMasker(mask_cutoff=self.plddt_threshold).parse_pdb(
                pdb_file
            )

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
            if enable_plddt_mask:
                mask = plddt_masks[new_seq.chain]
                # only structural sequence is masked here
                new_seq.structural_seq = mask.masked(new_seq.structural_seq)
            seq_dict.seqs.update({new_seq.chain: new_seq})

        if self.drop_fold_results_after_initialized:
            seq_dict.foldseek_results = None

        return (
            seq_dict
            if chain_ids is None or len(chain_ids) == 0
            else seq_dict.filtered(chains=tuple(chain_ids))
        )

    def parallel_queries(
        self,
        pdb_files: Union[tuple, list],
        enable_plddt_masks: Union[tuple[bool], bool] = False,
    ) -> tuple[StructuralAwareSequences]:
        if not isinstance(enable_plddt_masks, (tuple, list)):
            if not isinstance(enable_plddt_masks, bool):
                raise TypeError(
                    "enable_plddt_masks must be either a bool or a tuple of bools"
                )

            warnings.warn(
                PlddtMaskWarning(
                    f"Apply PLDDT mask ({enable_plddt_masks}) to all PDB queries"
                )
            )
            warnings.filterwarnings("ignore", category=PlddtMaskWarning)
            enable_plddt_masks = [enable_plddt_masks for i, v in enumerate(pdb_files)]

        if (l1 := len(enable_plddt_masks)) != (l2 := len(pdb_files)):
            raise ValueError(
                f"The length of plddt_masks ({l1}) must be equal to the length of pdb_files ({l2})"
            )
        return tuple(
            Parallel(n_jobs=self.nproc)(
                delayed(self.query)(pdb_file, None, plddt_mask)
                for pdb_file, plddt_mask in track(
                    zip(pdb_files, enable_plddt_masks), description="FoldSeeking..."
                )
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

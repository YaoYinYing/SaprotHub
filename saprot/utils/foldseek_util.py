from dataclasses import dataclass, field
import os
import platform
import subprocess
import tarfile
import time
import json
from typing import Literal, Sequence, Union
import numpy as np
import re
import sys
import pooch
import platformdirs

from saprot.utils.mask_tool import shorter_range

FOLDSEEK_STRUC_VOCAB = "pynwrqhgdlvtmfsaeikc#"
FOLDSEEK_STRUC_VOCAB_HINT = Literal['p', 'y', 'n', 'w', 'r', 'q', 'h', 'g', 'd', 'l', 'v', 't', 'm', 'f', 's', 'a', 'e', 'i', 'k', 'c', '#']

@dataclass
class Mask:
    mask_pos_range: str # set to None or '' to mask with full length.

    mask_label: str= '#'
    mask_separator: str = ","
    mask_connector: str = "-"

    zero_indexed:bool=False


    def update_from_masked_sequence(self, masked_sequence):
        mask: tuple[int]=tuple(i for i, k in enumerate(masked_sequence) if k == self.mask_label)
        self.mask_pos_range = shorter_range(mask, connector=self.mask_connector, seperator=self.mask_separator)
        print(f'Generate mask from masked sequence: {self.mask_pos_range}')

        self.zero_indexed=True




    def __post_init__(self):
        if len(self.mask_label) !=1:
            raise ValueError("mask_label must be a single character")



    @property
    def mask(self)->list[int]:
        if  self.mask_pos_range == '':
            raise ValueError('Mask pos range cannot be empty!')
        
        from saprot.utils.mask_tool import expand_range

        expanded_mask_pos=expand_range(shortened_str=self.mask_pos_range, connector=self.mask_connector, seperator=self.mask_separator)
        if not self.zero_indexed:
            expanded_mask_pos=[i-1 for i in expanded_mask_pos]

        return expanded_mask_pos
    

    def masked(self, sequence:Union[str, list,tuple])-> str:
        """
        Returns a masked sequence.
        """
        sequence=list(sequence)
        if len(sequence)<max(self.mask)+1:
            raise ValueError(f"Sequence length {len(sequence)} is less than the mask position range {self.mask_pos_range}")

        if self.mask_pos_range is None or self.mask_pos_range == '':
            self.mask_pos_range = f'1-{len(sequence)}'

            print(f'Mask is set to full length: {self.mask_pos_range}')

        for pos in self.mask:
            sequence[pos]=self.mask_label
        
        return ''.join(sequence)



        

@dataclass
class StrucuralSequence:
    seq:str
    struc_seq: Sequence[FOLDSEEK_STRUC_VOCAB_HINT]
    desc: str

    name: str
    name_chain: str=None
    chain: str=None


    def __post_init__(self):
        self.seq=self.seq.strip()
        self.struc_seq=self.struc_seq.strip().lower()
        self.name_chain=self.desc.split(" ")[0]
        self.chain=self.name_chain.replace(self.name, "").split("_")[-1]
        if len(self.seq) != len(self.struc_seq):
            raise ValueError("Sequence and structure sequence must be of the same length")
        
        if self.name.endswith('.cif') or self.name.endswith('.pdb'):
            self.name = self.name[:-4]

    @property
    def combined_sequence(self) -> str:
        combined_sequence=''.join(f'{_seq}{_struc_seq}' for _seq, _struc_seq in zip(self.seq, self.struc_seq.lower()))
        return combined_sequence
    

    def masked_seq(self, mask: Mask):
        return mask.masked(self.seq)
    

    def masked_struct_seq(self, mask: Mask):
        return mask.masked(self.struc_seq)



        

@dataclass
class StrucuralSequences:
    source_structure: str=None
    seqs:dict[str, StrucuralSequence]=field(default_factory=dict)
    

    def filtered(self, chains: tuple[str]) -> 'StrucuralSequences':
        return StrucuralSequences(
            source_structure=self.source_structure,
            seqs={
                chain: seq
                for chain, seq in self.seqs.items()
                if chain in chains
            }
        )
    
    def __getitem__(self,chain_id: str):
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
    base_url: str = 'https://mmseqs.com/foldseek/'
    

    # Gather system information
    uname = platform.uname()
    cpu_info = subprocess.run(['cat', '/proc/cpuinfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf8')

    arch: str = uname.machine

    os_build: str = uname.system

    def __post_init__(self):
        """
        Post-initialization method that sets up the FoldSeek binary.
        """

        if not os.path.exists(self.bin_dir):
            os.makedirs(self.bin_dir)
        

    @property
    def  bin_path(self) -> str: 
        return os.path.join(self.bin_dir, 'foldseek')

    def build_with_flag(self, flag: Literal['avx2', 'sse2']) -> bool:
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
        if self.os_build == 'Darwin':
            return f'{self.base_url}/foldseek-osx-universal.tar.gz'
        if self.os_build == 'Linux':
            if self.arch == 'x86_64':
                if self.build_with_flag('sse2'):
                    return f'{self.base_url}/foldseek-linux-sse2.tar.gz'
                if self.build_with_flag('avx2'):
                    return f'{self.base_url}/foldseek-linux-avx2.tar.gz'
            if self.arch == 'aarch64':
                return f'{self.base_url}/foldseek-linux-arm64.tar.gz'
        raise NotImplementedError(f'Unsupported platform {self.os_build} or architecture {self.arch}')

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
        compressed_file_path = pooch.retrieve(self.retrieve_url, known_hash=None, progressbar=True)
        with tarfile.open(compressed_file_path, 'r:gz') as tar:
            p=platformdirs.user_cache_dir('FoldSeek')
            tar.extractall(path=p)

            os.rename(os.path.join(p,'foldseek','bin', 'foldseek'), self.bin_path)
        
        os.remove(compressed_file_path)
        if not os.path.exists(self.bin_path):
            raise RuntimeError('Could not retrieve foldseek binary')
        
        # chmod +x
        os.chmod(self.bin_path, 0o555)


# Get structural seqs from pdb file
def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = False,
                  plddt_threshold: float = 70.) -> StrucuralSequences:
    """
    
    Args:
        foldseek: Binary executable file of foldseek
        path: Path to pdb file
        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
        process_id: Process ID for temporary files. This is used for parallel processing.
        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.
        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"Pdb file not found: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    name = os.path.basename(path)
    seq_dict: StrucuralSequences =StrucuralSequences(source_structure=path, seqs={})
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]

            new_seq=StrucuralSequence(seq=seq, struc_seq=struc_seq,desc=desc,name=name)
            
            # Mask low plddt
            if plddt_mask:
                plddts = extract_plddt(path)
                assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                
                # Mask regions with plddt < threshold
                indices = np.where(plddts < plddt_threshold)[0]
                np_seq = np.array(list(new_seq.struc_seq))
                np_seq[indices] = "#"
                new_seq.struc_seq = "".join(np_seq)
            
            seq_dict.seqs.update({new_seq.chain: new_seq})

        
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict if chains is not None or len(chains) == 0 else seq_dict.filtered(chains=tuple(chains))


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
            line = re.sub(' +', ' ', line).strip()
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
    

if __name__ == '__main__':
    foldseek = "/sujin/bin/foldseek"
    # test_path = "/sujin/Datasets/PDB/all/6xtd.cif"
    test_path = "/sujin/Datasets/FLIP/meltome/af2_structures/A0A061ACX4.pdb"
    plddt_path = "/sujin/Datasets/FLIP/meltome/af2_plddts/A0A061ACX4.json"
    res = get_struc_seq(foldseek, test_path, plddt_path=plddt_path, plddt_threshold=70.)
    print(res["A"][1].lower())
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from saprot.utils.mask_tool import shorter_range


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
        return self.structural_seq.strip("#") == ""

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
            seqs={
                chain: seq
                for chain, seq in self.seqs.items()
                if chain in chains
            },
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
class UniProtID:
    uniprot_id: str
    uniprot_type: Literal["AF2", "PDB"]
    chain_id: str

    SA_seq: StructuralAwareSequences = None

    def __post_init__(self):
        if self.uniprot_type not in ["AF2", "PDB"]:
            raise ValueError(
                f"UniProt type must be either 'AF2' or 'PDB', got {self.uniprot_type}"
            )

        if self.uniprot_id.endswith(".pdb") or self.uniprot_id.endswith(
            ".cif"
        ):
            self.uniprot_id = self.uniprot_id[:-4]

    @property
    def SA(self) -> StructuralAwareSequence:
        return self.SA_seq[self.chain_id]

    def set_sa_name(self, name):
        for sa in self.SA_seq.seqs.values():
            sa.name = name
    
    @property
    def is_AF2_structure(self) -> bool:
        return self.uniprot_type=="AF2"

@dataclass
class StructuralAwareSequencePair:
    seq_1: StructuralAwareSequence
    seq_2: StructuralAwareSequence

    @property
    def paired_sa(self):
        raise NotImplementedError

    @property
    def as_tuple(self) -> tuple[StructuralAwareSequence]:
        return (self.seq_1, self.seq_2)


@dataclass
class UniProtIDs:
    uniprot_ids: Union[tuple[UniProtID]]

    def __post_init__(self):
        if isinstance(self.uniprot_ids, UniProtID):
            self.uniprot_ids = [self.uniprot_ids]

    @property
    def all_labels(self) -> tuple[str]:
        return tuple(uniprot.uniprot_id for uniprot in self.uniprot_ids)

    def map_sa_to_uniprot_ids(self, sa_seqs: tuple[StructuralAwareSequences]):
        if not len(sa_seqs) == len(self.uniprot_ids):
            raise ValueError(
                f"The number of uniprot ids ({len(self.uniprot_ids)}) and the number of sa_seqs ({len(sa_seqs)}) must be equal."
            )

        for i, sa in enumerate(sa_seqs):
            self.uniprot_ids[i].SA_seq = sa
        return

    @property
    def SA_seqs_as_tuple(self) -> tuple[StructuralAwareSequence]:
        return tuple(x.SA for x in self.uniprot_ids)

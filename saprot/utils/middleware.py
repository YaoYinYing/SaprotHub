from dataclasses import dataclass
from typing import Protocol, Literal
import warnings

from saprot.utils.constants import DATA_TYPES_HINT, DATASET_TYPE2SEQ_TYPE


class MiddlewareWarning(Warning):
    ...


class Model(Protocol):
    training_data_type: Literal["AA", "SA"]


class StructuralAwareSequence(Protocol):
    amino_acid_seq: str
    combined_sequence: str
    _blind: bool


@dataclass
class SAFitter:
    model: Model
    dataset_source: DATA_TYPES_HINT

    def __post_init__(self):
        if (
            self.model.training_data_type == "SA"
            and DATASET_TYPE2SEQ_TYPE.get(self.dataset_source) != "SA"
        ):
            raise ValueError(
                f"{self.dataset_source} is not a SA dataset, but the model is SA-trained"
            )

    def __call__(self, input: StructuralAwareSequence) -> str:
        # print(f"Using {self.model.training_data_type} as input sequence.")
        if self.model.training_data_type == "AA":
            return input.amino_acid_seq

        if input._blind:
            warnings.warn(
                MiddlewareWarning(
                    "Model is trained on Structural-Awared data, but input is blind, meaning that a full masked structure sequence is used."
                )
            )
            warnings.filterwarnings(
                "ignore", category=MiddlewareWarning
            )  # ignore the warning once it's been shown
        return input.combined_sequence

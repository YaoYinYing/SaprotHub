from dataclasses import dataclass
from typing import Any, Protocol, Literal, Generator, Union
from typing_extensions import runtime_checkable
import warnings

from saprot.utils.constants import DATA_TYPES_HINT, DATASET_TYPE2SEQ_TYPE


class MiddlewareWarning(Warning):
    ...


class Model(Protocol):
    training_data_type: Literal["AA", "SA"]


@runtime_checkable
class StructuralAwareSequence(Protocol):
    amino_acid_seq: str
    combined_sequence: str
    _blind: bool


@runtime_checkable
class StructuralAwareSequencePair(Protocol):
    seq_1: StructuralAwareSequence
    seq_2: StructuralAwareSequence

    as_tuple: tuple[StructuralAwareSequence]


@dataclass
class SADataAdapter:
    """
    Adapter class to combine a model with its data source, providing middleware for data processing.

    Parameters:
    - model: Model type, representing a trained model.
    - dataset_source: Source of the dataset, indicating the origin of the data, typed as DATA_TYPES_HINT.
    """

    model: Model
    dataset_source: DATA_TYPES_HINT

    def __post_init__(self):
        """
        Post-initialization check to ensure compatibility between model training data type and dataset source.
        Raises ValueError if the model is trained on 'SA' data but the dataset source is not compatible.
        """
        if (
            self.model.training_data_type == "SA"
            and DATASET_TYPE2SEQ_TYPE.get(self.dataset_source) != "SA"
        ):
            raise ValueError(
                f"{self.dataset_source} is not a SA dataset, but the model is SA-trained"
            )

    def _adapt_single_SA(self, input: StructuralAwareSequence) -> str:
        """
        Adapts a single StructuralAwareSequence object into a string format based on the model's training data type.

        Parameters:
        - input: A StructuralAwareSequence object to be adapted.

        Returns:
        - str: Adapted sequence as a string.
        """
        if self.model.training_data_type == "AA":
            return input.amino_acid_seq

        if input._blind:
            warnings.warn(
                MiddlewareWarning(
                    "Model is trained on Structural-Aware data, but input is blind, meaning that a full masked structure sequence is used."
                )
            )
            warnings.filterwarnings(
                "ignore", category=MiddlewareWarning
            )  # Ignore the warning once it's been shown
        return input.combined_sequence

    def _adapt_single_SAP(
        self, input: StructuralAwareSequencePair
    ) -> tuple[str]:
        """
        Adapts a single StructuralAwareSequencePair into a tuple of strings.

        Parameters:
        - input: A StructuralAwareSequencePair object to be adapted.

        Returns:
        - tuple[str]: Adapted sequences as a tuple of strings.
        """
        return tuple(self._adapt_single_SA(sa) for sa in input.as_tuple)

    def _adapt_single(
        self,
        input: Union[
            StructuralAwareSequence, StructuralAwareSequencePair, Any
        ],
    ) -> Union[str, tuple[str]]:
        """
        General adapter method to handle both single sequences and pairs.

        Parameters:
        - input: An instance of StructuralAwareSequence, StructuralAwareSequencePair, or any other type.

        Returns:
        - Union[str, tuple[str]]: Adapted output as either a string or a tuple of strings.

        Raises:
        - ValueError: If the input type is not supported.
        """
        if isinstance(input, StructuralAwareSequence):
            return self._adapt_single_SA(input)
        if isinstance(input, StructuralAwareSequencePair):
            return self._adapt_single_SAP(input)
        raise ValueError(f"Input type {type(input)} not supported.")

    def adapt(
        self,
        inputs: Union[
            tuple[StructuralAwareSequence], tuple[StructuralAwareSequencePair]
        ],
    ) -> Generator:
        """
        Adapts a collection of inputs, yielding each adapted item.

        Parameters:
        - inputs: A tuple of StructuralAwareSequence or StructuralAwareSequencePair objects.

        Returns:
        - Generator: A generator yielding adapted items.
        """
        for input in inputs:
            yield self._adapt_single(input)

    def __call__(
        self,
        inputs: Union[
            tuple[StructuralAwareSequence], tuple[StructuralAwareSequencePair]
        ],
    ) -> Generator:
        """
        Allows the adapter to be called directly as a function.

        Parameters:
        - inputs: A tuple of StructuralAwareSequence or StructuralAwareSequencePair objects.

        Returns:
        - Generator: A generator yielding adapted items.
        """
        return self.adapt(inputs)

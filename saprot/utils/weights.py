import os
from typing import Literal
from dataclasses import dataclass

import platformdirs
from huggingface_hub import snapshot_download


SaProtModelHint = Literal[
    "SaProt_35M_AF2",
    "SaProt_650M_PDB",
    "SaProt_650M_AF2",
    "SaProt_650M_AF2_inverse_folding",
    "SaProt_35M_AF2_seqOnly",
    "esmfold_v1",
]
MODEL_LOADER_TYPE_HINT = Literal["native", "esm", "esmfold"]
MODEL_LOADER_TYPE = (
    "native",
    "esm",
    "esmfold",
)


@dataclass
class PretrainedModel:
    """
    A class representing a pretrained model, providing functionality to fetch and load the model.

    Attributes:
        dir (str): Directory where the model is stored.
        model_name (SaProtModelHint): Name of the model to load.
        loader_type (MODEL_LOADER_TYPE_HINT, optional): Type of loader to use. Defaults to None.
        huggingface_id (str): Identifier for the model on HuggingFace. Set to None to use a local model. Defaults to 'westlake-repl'.
    """

    dir: str
    model_name: SaProtModelHint
    loader_type: MODEL_LOADER_TYPE_HINT = None
    huggingface_id: str = "westlake-repl"

    @property
    def weights_dir(self):
        """Returns the full path to the model's weights."""
        return os.path.join(self.dir, self.model_name)

    def __post_init__(self):
        """
        Initializes the model by fetching and loading it from either a remote repository or a local directory.
        Validates the HuggingFace ID and directory paths, and downloads the model if necessary.
        """
        if self.loader_type is None:
            self.loader_type = "native"

        if self.loader_type not in MODEL_LOADER_TYPE:
            raise ValueError(
                f"Invalid loader type: {self.loader_type}. Valid options are: {MODEL_LOADER_TYPE}"
            )

        if self.huggingface_id == "":
            print(
                "Ensure the correct path is set and you have permission to access the local model."
            )
            raise ValueError(f"Invalid HuggingFace ID: {self.huggingface_id}")

        if self.huggingface_id is None:
            print("Hugging Face ID is set to None, searching for a local model.")
            if not self.dir or not os.path.exists(self.dir):
                raise FileNotFoundError(f"File {self.dir} does not exist")
            return

        if self.dir is None:
            self.dir = platformdirs.user_cache_dir("SaProt")

        
        self.dir = os.path.abspath(self.dir)

        if self.model_name is None:
            raise ValueError("A model name must be specified to load")

        os.makedirs(self.dir, exist_ok=True)

        if not os.path.exists(self.weights_dir):
            print(
                f"Fetching model weights ({self.model_name}) from HuggingFace, which might take some time..."
            )
            try:
                print(f"Pretrained model weights will be saved to {self.dir}")
                self._fetch_model()
            except EnvironmentError:
                print(
                    "Failed to retrieve model weights from HuggingFace, cleaning up..."
                )
                import shutil

                shutil.rmtree(self.weights_dir)
                raise ConnectionError("Failed to retrieve model weights from HuggingFace")

    def _fetch_model(self, force_redownload=False):
        """
        Downloads the model from HuggingFace based on the model name and loader type.

        Args:
            force_redownload (bool, optional): Whether to force redownload of the model. Defaults to False.
        """
        download_args = {
            "repo_id": f"{self.huggingface_id}/{self.model_name}",
            "local_dir": self.weights_dir,
        }

        if self.loader_type in ["native", None]:
            download_args["ignore_patterns"] = ["*.pt"]
        else:
            download_args["allow_patterns"] = ["*.pt"]

        if force_redownload:
            download_args["force_download"] = True

        dir = snapshot_download(**download_args)
        print(f"Pretrained Model Weights have been downloaded to {dir}")

    def load_model(self):
        """
        Loads the model weights from the specified directory based on the loader type.

        Returns:
            tuple: Depending on the loader type, returns a tuple containing the model and tokenizer.
        """
        print(f"Loading model weights from {self.weights_dir}")

        if self.loader_type in [None, "native"]:
            from transformers import EsmTokenizer, EsmForMaskedLM

            tokenizer = EsmTokenizer.from_pretrained(self.weights_dir)
            model = EsmForMaskedLM.from_pretrained(self.weights_dir)
            return model, tokenizer
        if self.loader_type == "esm":
            from saprot.utils.esm_loader import load_esm_saprot

            model, alphabet = load_esm_saprot(
                os.path.join(self.weights_dir, f"{self.model_name}.pt")
            )
            return model, alphabet
        if self.loader_type == "esmfold":
            from transformers import AutoTokenizer, EsmForProteinFolding

            if self.huggingface_id != "facebook":
                raise ValueError(
                    f"HuggingFace ID must be 'facebook' for ESMFold, received {self.huggingface_id}"
                )
            esmfoldtokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            return esmfold, esmfoldtokenizer

        raise TypeError(
            f"Loader must be one of {MODEL_LOADER_TYPE} but received `{self.loader_type}`"
        )

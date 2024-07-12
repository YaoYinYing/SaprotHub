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
]
ModelLoaderType = Literal["native", "esm"]


@dataclass
class PretrainedModel:
    """
    A class representing a pretrained model with functionality to fetch and load the model.

    Attributes:
        dir (str): The directory where the model will be stored.
    """

    dir: str
    model_name: SaProtModelHint
    loader_type: ModelLoaderType = None

    huggingface_id: str='westlake-repl' # set to None to use local model

    @property
    def weights_dir(self):
        return os.path.join(self.dir, self.model_name)

    def __post_init__(self):
        """
        Initializes the model by fetching and loading it.
        """
        if self.huggingface_id =='':
            print(f'If you are using a local model, make sure you have the right path set and permission to access it.')
            raise ValueError(f'Invalid huggingface id: {self.huggingface_id}')
        
        if self.huggingface_id is None:
            print('hugging face id is set to None, looking for local model')
            if not self.dir or not os.path.exists(self.dir):
                raise FileNotFoundError(f'File {self.dir} does not exist')
            return
        
        
        if self.dir is None:
            self.dir = platformdirs.user_cache_dir("SaProt")
        
        print(f'Pretrained model weights will be saved to {self.dir}')

        self.dir = os.path.abspath(self.dir)

        if self.model_name is None:
            raise ValueError("One must specify a model to load")
        

        os.makedirs(self.dir, exist_ok=True)


        if not os.path.exists(self.weights_dir):
            print(
                f"Retrieving model weights ({self.model_name}) from HuggingFace, this may take a while..."
            )
            
            try:
                self._fetch_model(self.model_name)
            except EnvironmentError as e:
                print(
                    f"Failed to retrieve model weights from HuggingFace, removing..."
                )
                import shutil
                
                shutil.rmtree(self.weights_dir)
                raise RuntimeError("Failed to retrieve model weights from HuggingFace")

    def _fetch_model(self, force_redownload=False) -> str:
        """
        Fetches the specified model and downloads it to the designated directory.

        Args:
            model (SaProtModelHint): The hint for identifying which model to fetch.
            loader_type (ModelLoaderType, optional): The type of loader to use. Defaults to None.

        Returns:
            str: The directory where the model was downloaded.
        """

        download_args = {
            "repo_id": f"{self.huggingface_id}/{self.model_name}",
            "local_dir": self.weights_dir,
        }

        if self.loader_type == "native" or self.loader_type is None:
            # Download all files except the .pt file
            download_args["ignore_patterns"] = ["*.pt"]
        else:
            # Download only the .pt file
            download_args["allow_patterns"] = ["*.pt"]

        if force_redownload:
            download_args["force_download"] = True

        dir = snapshot_download(**download_args)

        print(f"Pretrained Model Weights is downloaded to {dir}")

    @property
    def tokenizer(self):
        from transformers import EsmTokenizer

        return EsmTokenizer.from_pretrained(self.weights_dir)

    @property
    def model(self):
        from transformers import EsmForMaskedLM

        return EsmForMaskedLM.from_pretrained(self.weights_dir)

    def load_model(self, loader_type: ModelLoaderType = None):
        """
        Loads the specified model from the given directory.

        Args:
            dir (str, optional): The directory from which to load the model. If None, uses the cache directory. Defaults to None.
            model_name (SaProtModelHint, optional): The name of the model to load. Must be specified. Defaults to None.
            loader_type (ModelLoaderType, optional): The type of loader to use. Defaults to None.

        Raises:
            ValueError: If no model is specified.

        Returns:
            tuple: Depending on the loader_type, returns either a tuple of (tokenizer, model) or (model, alphabet).
        """

        print(f"Loading model weights from {self.weights_dir}")

        if loader_type is None or loader_type == "native":

            return self.model, self.tokenizer
        else:
            from saprot.utils.esm_loader import load_esm_saprot

            model, alphabet = load_esm_saprot(
                os.path.join(self.weights_dir, f"{self.model_name}.pt")
            )
            return model, alphabet

from dataclasses import dataclass
import os
from typing import Literal, Protocol

import pooch
from joblib import Parallel, delayed
from rich.progress import track


# Define a protocol class to specify the required attributes of UniProt identifiers
class UniProtID(Protocol):
    uniprot_id: str
    uniprot_type: Literal["AF2", "PDB"]
    chain_id: str


@dataclass
class StructureDownloader:
    """
    A class to download protein structure files from AlphaFold or RCSB PDB.

    Attributes:
        data_type (Literal): The type of data to download ('pdb', 'mmcif', 'plddt', 'pae').
        save_dir (str): The directory where the downloaded files will be saved.
        nproc (int): Number of processes to use for parallel downloading.
    """
    data_type: Literal['pdb', 'mmcif', 'plddt', 'pae']
    save_dir: str
    nproc: int = os.cpu_count()

    def process_url(self, data_id: UniProtID) -> str:
        """
        Processes the URL and downloads the file based on the UniProt identifier.

        Args:
            data_id (UniProtID): An object containing UniProt identifier details.

        Returns:
            str: The path to the downloaded file.
        """
        if data_id.uniprot_type == 'PDB' and (self.data_type == 'plddt' or self.data_type == 'pae'):
            raise ValueError(f"{self.data_type} is not available for PDB")

        # Determine the base URL and filename format based on UniProt type
        if data_id.uniprot_type == 'AF2':
            url_dict = {
                "pdb": "AF-{}-F1-model_v4.pdb",
                "mmcif": "AF-{}-F1-model_v4.cif",
                "plddt": "AF-{}-F1-confidence_v4.json",
                "pae": "AF-{}-F1-predicted_aligned_error_v4.json"
            }
            save_dict = {
                "pdb": "{}.pdb",
                "mmcif": "{}.cif",
                "plddt": "{}.json",
                "pae": "{}.json"
            }
            base_url = 'https://alphafold.ebi.ac.uk/files/'
            filename = url_dict[self.data_type].format(data_id.uniprot_id)
            file_save_name = save_dict[self.data_type].format(data_id.uniprot_id)
        else:
            save_dict = {
                "pdb": "{}.pdb",
                "mmcif": "{}.cif"
            }
            base_url = 'https://files.rcsb.org/download/'
            filename = save_dict[self.data_type].format(data_id.uniprot_id)
            file_save_name = filename

        # Download the file using the provided URL
        return pooch.retrieve(
            url=os.path.join(base_url, filename),
            known_hash=None,
            path=self.save_dir,
            progressbar=False,
            fname=file_save_name if file_save_name is not None else filename,
        )

    def run(self, payload: list[UniProtID]):
        """
        Runs the download process in parallel for a list of UniProt identifiers.

        Args:
            payload (list[UniProtID]): A list of UniProt identifier objects.

        Returns:
            list[str]: A list of paths to the downloaded files.
        """
        # Use joblib's Parallel and delayed to run process_url in parallel
        return Parallel(n_jobs=self.nproc)(
            delayed(self.process_url)(data_id) for data_id in track(payload)
        )
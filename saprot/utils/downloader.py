from dataclasses import dataclass
import os
import pooch
from joblib import Parallel, delayed
from rich.progress import track

@dataclass
class PoochDownloader:
    base_url: str
    save_path: str
    nproc: int = os.cpu_count()

    def __post_init__(self):
        if not self.save_path:
            raise ValueError("save_path must be specified")
        os.makedirs(self.save_path, exist_ok=True)

    def run(self, filename: str, saved_filename: str=None):
        return pooch.retrieve(
            url=os.path.join(self.base_url, filename),
            known_hash=None,
            path=self.save_path,
            progressbar=False,
            fname=saved_filename if saved_filename is not None else filename,
        )

    def cocurrent_download(self, filenames: list[str], saved_filenames: list[str]=None):
        if not saved_filenames:
            saved_filenames = filenames
            
        return Parallel(n_jobs=self.nproc)(
            delayed(self.run)(filename,saved_filename) for filename,saved_filename in track(zip(filenames,saved_filenames))
        )

class AlphaDBDownloader:
    """
    Download files from AlphaFold2 database.
    """

    def __init__(self, uniprot_ids, type: str, save_dir: str, **kwargs):
        url_dict = {
            "pdb": "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb",
            "mmcif": "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.cif",
            "plddt": "https://alphafold.ebi.ac.uk/files/AF-{}-F1-confidence_v4.json",
            "pae": "https://alphafold.ebi.ac.uk/files/AF-{}-F1-predicted_aligned_error_v4.json"
        }

        save_dict = {
            "pdb": "{}.pdb",
            "mmcif": "{}.cif",
            "plddt": "{}.json",
            "pae": "{}.json"
        }

        url = url_dict[type]
        base_url=os.path.dirname(url)
        file_base_name=os.path.basename(url)

        self.downloader = PoochDownloader(base_url=base_url, save_path=save_dir, **kwargs)
        self.filenames = [file_base_name.format(uniprot) for uniprot in uniprot_ids]
        self.saved_filenames=[save_dict[type].format(uniprot) for uniprot in uniprot_ids]

    def run(self):
        self.downloader.cocurrent_download(self.filenames, self.saved_filenames)

class PDBDownloader:
    """
    Download files from PDB.
    """

    def __init__(self, pdb_ids, type: str, save_dir: str, **kwargs):


        save_dict = {
            "pdb": "{}.pdb",
            "mmcif": "{}.cif"
        }

        base_url='https://files.rcsb.org/download/'

        self.downloader = PoochDownloader(base_url=base_url, save_path=save_dir, **kwargs)
        self.filenames = [save_dict[type].format(pdb_id) for pdb_id in pdb_ids]
        self.saved_filenames=self.filenames

    def run(self):
        self.downloader.cocurrent_download(self.filenames)

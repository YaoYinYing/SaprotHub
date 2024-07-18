import unittest
from unittest.mock import patch, MagicMock
import os

from saprot.utils.downloader import AlphaDBDownloader, PDBDownloader, PoochDownloader

class TestPoochDownloader(unittest.TestCase):

    @patch("os.makedirs")
    def test_post_init_creates_directory(self, mock_makedirs):
        downloader = PoochDownloader(base_url="https://alphafold.ebi.ac.uk/files/AF-Q8W3K0-F1-model_v4.cif", save_path="output/downloader/pooch_1")
        mock_makedirs.assert_called_with("output/downloader/pooch_1", exist_ok=True)

    @patch("pooch.retrieve")
    def test_run_calls_pooch_retrieve(self, mock_retrieve):
        downloader = PoochDownloader(base_url="https://alphafold.ebi.ac.uk/files/", save_path="output/downloader/pooch_2")
        downloader.run("AF-Q8W3K0-F1-model_v4.cif")
        mock_retrieve.assert_called_with(
            url="https://alphafold.ebi.ac.uk/files/AF-O15552-F1-predicted_aligned_error_v4.json",
            known_hash=None,
            path="output/downloader/pooch_2",
            progressbar=False
        )

    @patch("pooch.retrieve")
    @patch("joblib.Parallel")
    def test_concurrent_download_calls_parallel(self, mock_parallel, mock_retrieve):
        downloader = PoochDownloader(base_url="https://alphafold.ebi.ac.uk/files", save_path="output/downloader/pooch_cocurrent")
        filenames = ["AF-P0ADB1-F1-model_v4.pdb", "AF-P0ADB1-F1-model_v4.cif", "AF-P0ADB1-F1-predicted_aligned_error_v4.json"]
        received=downloader.cocurrent_download(filenames)
        mock_parallel.assert_called()

class TestAlphaDBDownloader(unittest.TestCase):

    @patch.object(PoochDownloader, 'cocurrent_download')
    def test_run_calls_cocurrent_download(self, mock_cocurrent_download):
        downloader = AlphaDBDownloader(uniprot_ids=["P76011", "Q5VSL9"], type="pdb", save_dir="output/downloader/AF2DB")
        downloader.run()
        expected_filenames = ["P76011.pdb", "Q5VSL9.pdb"]
        mock_cocurrent_download.assert_called_with(expected_filenames)

class TestPDBDownloader(unittest.TestCase):

    @patch.object(PoochDownloader, 'cocurrent_download')
    def test_run_calls_cocurrent_download(self, mock_cocurrent_download):
        downloader = PDBDownloader(pdb_ids=["1UBQ", "8AC8"], type="pdb", save_dir="output/downloader/pdb")
        downloader.run()
        expected_filenames = ["1UBQ.pdb", "8AC8.pdb"]
        mock_cocurrent_download.assert_called_with(expected_filenames)

if __name__ == '__main__':
    unittest.main()

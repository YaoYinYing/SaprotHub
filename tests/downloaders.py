
import os

from saprot.utils.downloader import AlphaDBDownloader, PDBDownloader, PoochDownloader


def test_pooch():
    downloader = PoochDownloader(base_url="https://alphafold.ebi.ac.uk/files/AF-Q8W3K0-F1-model_v4.cif", save_path="output/downloader/pooch_1")
    #assert os.path.exists(downloader.save_path)

    downloader = PoochDownloader(base_url="https://alphafold.ebi.ac.uk/files/", save_path="output/downloader/pooch_2")
    downloader.run("AF-Q8W3K0-F1-model_v4.cif")

    #assert os.path.exists(os.path.join(downloader.save_path,'"AF-Q8W3K0-F1-model_v4.cif"'))


    downloader = PoochDownloader(base_url="https://alphafold.ebi.ac.uk/files", save_path="output/downloader/pooch_cocurrent")
    filenames = ["AF-P0ADB1-F1-model_v4.pdb", "AF-P0ADB1-F1-model_v4.cif", "AF-P0ADB1-F1-predicted_aligned_error_v4.json"]

    downloader.cocurrent_download(filenames)

def test_afdb_download():
    downloader = AlphaDBDownloader(uniprot_ids=["P76011", "Q5VSL9"], type="pdb", save_dir="output/downloader/AF2DB")
    downloader.run()
    expected_filenames = ["P76011.pdb", "Q5VSL9.pdb"]


def test_pdb_download():
    downloader = PDBDownloader(pdb_ids=["1UBQ", "8AC8"], type="pdb", save_dir="output/downloader/pdb")
    downloader.run()
    expected_filenames = ["1UBQ.pdb", "8AC8.pdb"]

def main():
    test_pooch()
    test_afdb_download()
    test_pdb_download()

if __name__ == '__main__':
    main()

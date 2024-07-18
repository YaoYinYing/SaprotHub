import requests
import json
from easydict import EasyDict


# Convert the pdb id to the uniprot id. For every matched uniprot id,
# we only select one of corresponding pdb chains to return.
def pdb2uniprot(pdb):
    """
    Args:
        pdb: pdb id.

    Returns:
        A list contains all matched uniprot ids and corresponding chains
    """
    response = requests.get(
        f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"
    )
    res = EasyDict(json.loads(response.text))

    results = []
    keys = set()
    for uniprot, info in res[pdb].UniProt.items():
        for chain in info.mappings:
            if chain.chain_id not in keys:
                results.append((f"{pdb}_{chain.chain_id}", uniprot))
                keys.add(chain.chain_id)

    return results


# Convert the pdb id with specific chain to the uniprot id
def chain2uniprot(pdb, chain):
    response = requests.get(
        f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"
    )
    res = EasyDict(json.loads(response.text))

    for uniprot, info in res[pdb].UniProt.items():
        for mappings in info.mappings:
            if mappings.chain_id == chain:
                return uniprot

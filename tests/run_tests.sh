#! bash

# make it  stop if error occurs
set -e

python tests/basic_load.py
python tests/mutation_effect.py
python tests/structure_awared.py
python tests/inverse_folding.py
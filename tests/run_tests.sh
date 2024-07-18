#! bash

# make it  stop if error occurs
set -e

# basic functions
python tests/basic/basic_load.py
python tests/basic/structure_awared.py
python tests/basic/call_tmalign.py
python tests/basic/dataset_dispatch.py


# direct predictions
python tests/direct_uses/mutation_effect.py
python tests/direct_uses/inverse_folding.py
# python tests/direct_uses/inverse_folding_refold.py
python tests/direct_uses/deep_mutagenese_scan.py

# tasks
python tests/tasks/regression-stability_aa.py
python tests/tasks/regression-thermalstability_sa.py
python tests/tasks/regression-thermalstability_sa_blind.py
python tests/tasks/classification-subcellular.py

python tests/tasks/pair_regression-similarity.py
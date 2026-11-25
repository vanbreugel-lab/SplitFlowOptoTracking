#!/usr/bin/env bash
# Smoke test: can every script and notebook in the repository find the data it asks for?
#
# Reads all the code without running it (seconds, no Gurobi needed), including the code that the
# per-figure scripts never run: fit scripts, simulations and hand-run notebooks. For every data
# lookup it checks that the key is in Data/data_paths.yaml and that the file is on disk, and it
# reports any hard-coded data path that bypasses data_paths.yaml.
#
# Usage:
#   RunFigures/check_code_data.sh              # whole repository; lists only the problems
#   RunFigures/check_code_data.sh -v           # list every lookup
#   RunFigures/check_code_data.sh FigureGeneration/Main/fig_5_unifying_algo_analysis   # one folder
#
# Exit code 1 on an unknown key, a missing required file or a hard-coded data path.
# See also Data/check_data.sh, which checks Data/ against the same list.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python "$RUN_DIR/check_code_data.py" "$@"

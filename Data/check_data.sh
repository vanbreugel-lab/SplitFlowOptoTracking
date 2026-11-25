#!/usr/bin/env bash
# Check that every data file the code uses is in this Data/ directory.
#
# The list of files is Data/data_paths.yaml. Run this after downloading the data, and after
# moving or renaming anything in Data/.
#
# Usage:
#   Data/check_data.sh              # is everything there?
#   Data/check_data.sh --unlisted   # also list files in Data/ that no code refers to
#   Data/check_data.sh --deep       # also open every file's header (slower)
#
# Exit code 1 if a required file or folder is missing. Files under NOT IN DATA in
# data_paths.yaml are listed but do not fail the check.
# See also RunFigures/check_code_data.sh, which checks the code against the same list.

source "$(dirname "${BASH_SOURCE[0]}")/../RunFigures/_common.sh"

python "$REPO_ROOT/Code/data_paths.py" check "$@"

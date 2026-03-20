#!/bin/bash

set -e

LABELS=("CFD_casting" "CFD_circling" "flash" "sham" "WT_flash" "WT_sham" "unifying")

for label in "${LABELS[@]}"; do
    echo "Running make_figure_align_course_directions.py --label $label"
    python make_figure_align_course_directions.py --label "$label"

    echo "Running make_figure_unifying_algo_results_slope_axis.py --label $label"
    python make_figure_unifying_algo_results_slope_axis.py --label "$label"
done

echo "All done."

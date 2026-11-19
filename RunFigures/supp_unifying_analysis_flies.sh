#!/usr/bin/env bash
# Supplemental: unifying analysis, all WT split-flow flies
# Runs the plotting notebooks for this figure (no data re-analysis).
# Final SVG: see figures.yaml (id: supp_unifying_analysis_flies).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

begin_figure supp_unifying_analysis_flies "Supplemental: unifying analysis, all WT split-flow flies"
D=FigureGeneration/Supplemental/supp_fig_11_unifying_flies_shams

# A and B draw one row (layers <label>_course / <label>_stats) per run; C draws one
# <condition>_mae panel per run. All write the same SVG, so run them in sequence.
for label in flash sham WT_flash WT_sham; do
    run_notebook "$D" A_figure_align_course_directions_flies_allwtsplitflow.ipynb FIG_LABEL=$label
done
for label in flash sham WT_flash WT_sham; do
    run_notebook "$D" B_figure_unifying_algo_results_allwtsplitflow.ipynb FIG_LABEL=$label
done
for condition in laminar stillair unsteady; do
    run_notebook "$D" C_MI_plots_flies_allwtsplitflow.ipynb FIG_CONDITION=$condition
done

end_figure

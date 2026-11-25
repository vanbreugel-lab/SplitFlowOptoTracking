# Blue crab (*Callinectes sapidus*) tracking an odour plume

## Source

Dickman BD (2008). Chemical and hydromechanical cue structure in the context of turbulent odor plume tracking. PhD thesis, Georgia Institute of Technology.

- Data repository: none; the trajectory was digitized by hand from a published figure
- Paper: `paper/Dickman_2008_PhD_thesis.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | screenshots of the appendix time-trace figures for crab 105N and crab 1271O (the thesis writes the second id with a letter O; the file names here use a zero): panel B shows streamwise velocity Vx and transverse velocity Vy against time Files: `raw/crab_105N.png`, `raw/crab_12710.png` |
| Hand-labeled data | hand-digitized Vx(t) and Vy(t) traces, one csv per trace; the notebook builds the path from the two velocity traces: `digitized/crab_105N_vx.csv`, `digitized/crab_105N_vy.csv`, `digitized/crab_12710_vx.csv`, `digitized/crab_12710_vy.csv` |
| Cleaned trajectory | `crab_105N_vx_trajec.parquet`, `crab_12710_trajec.parquet` |
| Unifying analysis | `crab_105N_vx_unifying_analysis.parquet`, `crab_12710_unifying_analysis.parquet` |
| Notebook | `crab_new_105N.ipynb`, `crab_new_12710.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/crabs/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

## Notes

- Only `crab_12710` is in `df_all_species.parquet` (used in the figures).

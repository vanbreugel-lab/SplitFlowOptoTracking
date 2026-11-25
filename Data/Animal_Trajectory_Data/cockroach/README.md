# Walking cockroach (*Periplaneta americana*) tracking a pheromone plume

## Source

Talley JL, White EB, Willis MA (2023). A comparison of odor plume-tracking behavior of walking and flying insects in different turbulent environments. *Journal of Experimental Biology*.

- DOI: https://doi.org/10.1242/jeb.244254
- Data repository: Dryad: https://doi.org/10.5061/dryad.m63xsj463
- Paper: `paper/Talley_2023_JExpBiol.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | the authors' tracking files from Dryad, `raw/wind100/laminar/*.dat` (24 trials) and `raw/wind100/turbulent/*.dat` (20 trials); whitespace-separated columns time, x, y, ... |
| Hand-labeled data | none; the raw data are tracked positions |
| Cleaned trajectory | `23april09i_trajec.parquet` |
| Unifying analysis | `23april09i_unifying_analysis.parquet` |
| Notebook | `cockroach_laminar.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/cockroach/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

## Notes

- The trial used in the paper is `raw/wind100/laminar/23april09i.dat`.
- `unused_fits/` holds earlier fits (18March09I and an older 23april09i) that were stored next to the raw files. No figure reads them.

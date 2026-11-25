# Flying moth (*Manduca sexta*) tracking a pheromone plume

## Source

Talley JL, White EB, Willis MA (2023). A comparison of odor plume-tracking behavior of walking and flying insects in different turbulent environments. *Journal of Experimental Biology*.

- DOI: https://doi.org/10.1242/jeb.244254
- Data repository: Dryad: https://doi.org/10.5061/dryad.m63xsj463
- Paper: `paper/Talley_2023_JExpBiol.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | the authors' tracking files from Dryad, `raw/wind100/{laminar,turbulent}/` and `raw/wind25/{laminar,turbulent}/` (`*.dat`, whitespace-separated columns time, x, y, z, ...), and their MATLAB reader `raw/tracks.m` |
| Hand-labeled data | none; the raw data are tracked positions |
| Cleaned trajectory | `1Oct08B_trajec.parquet`, `1Oct08E_trajec.parquet` |
| Unifying analysis | `1Oct08B_unifying_analysis.parquet`, `1Oct08E_unifying_analysis.parquet` |
| Notebook | `moth_1Oct08B.ipynb`, `moth_1Oct08E.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/moths/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

## Notes

- The trials used in the paper are `raw/wind100/laminar/1Oct08B.3.dat` and `raw/wind100/turbulent/1Oct08E.3.dat`.

# Flying mosquito (*Aedes aegypti*) in a wind tunnel

## Source

van Breugel F, Riffell J, Fairhall A, Dickinson MH (2015). Mosquitoes use vision to associate odor plumes with thermal targets. *Current Biology* 25:2123-2129.

- DOI: https://doi.org/10.1016/j.cub.2015.06.046
- Data repository: none; the trajectory was digitized by hand from a published figure
- Paper: `paper/vanBreugel_2015_CurrBiol.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | screenshot of an example flight trajectory from Fig. 1 (wind tunnel, 1.2 m long; the other axis is labelled .33 m) Files: `raw/mosquito_trajec.png` |
| Hand-labeled data | hand-digitized x, y points of the trajectory: `digitized/mosquito_trajec.csv` |
| Cleaned trajectory | `mosquito_trajec_trajec.parquet` |
| Unifying analysis | `mosquito_trajec_unifying_analysis.parquet` |
| Notebook | `mosquito_new.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/mosquitoes/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

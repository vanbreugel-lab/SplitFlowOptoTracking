# Freshwater eel (*Anguilla australis*, *A. dieffenbachii*) searching for a food odour in flowing water

## Source

Carton AG, Montgomery JC (2003). Evidence of a rheotactic component in the odour search behaviour of freshwater eels. *Journal of Fish Biology*.

- DOI: https://doi.org/10.1046/j.1095-8649.2003.00015.x
- Data repository: none; the trajectory was digitized by hand from a published figure
- Paper: `paper/Carton_2003_JFishBiol.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | screenshot of panel (b), "Eel 7"; axes in metres Files: `raw/eel.png` |
| Hand-labeled data | hand-digitized x, y points of the trajectory: `digitized/eel.csv` |
| Cleaned trajectory | `eel_trajec.parquet` |
| Unifying analysis | `eel_unifying_analysis.parquet` |
| Notebook | `eel_new.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/eels/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

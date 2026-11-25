# *Nautilus pompilius* tracking an odour plume in a flume

## Source

Basil JA, Hanlon RT, Sheikh SI, Atema J (2000). Three-dimensional odor tracking by *Nautilus pompilius*. *Journal of Experimental Biology* 203:1409-1414.

- Data repository: none; the trajectory was digitized by hand from a published figure
- Paper: `paper/Basil_2000_JExpBiol.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | screenshot of Fig. 1C (tracks plotted against distance from the source, 0-200 cm); the yellow track was digitized Files: `raw/nautilus_yellow.png` |
| Hand-labeled data | hand-digitized x, y points of the trajectory: `digitized/nautilus_yellow.csv` |
| Cleaned trajectory | `nautilus_yellow_trajec.parquet` |
| Unifying analysis | `nautilus_yellow_unifying_analysis.parquet` |
| Notebook | `nautilus_yellow_new.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/nautilus/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

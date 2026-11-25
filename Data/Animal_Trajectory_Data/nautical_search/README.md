# Search-and-rescue vessels executing standard search patterns

## Source

Varlamis I, Tserpes K, Sardianos C. Detecting Search and Rescue missions from AIS data. Conference paper (the PDF carries no venue or year; its file name indicates ICDE). AIS data from July-September 2015.

- Data repository: none; the trajectory was digitized by hand from a published figure
- Paper: `paper/Varlamis_Tserpes_Sardianos_SAR_from_AIS.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | screenshots of Fig. 5a (creeping line search; `charlie_sierra`) and Fig. 4b (expanding square search; `sierra_sierra`); axes are latitude / longitude Files: `raw/charlie_sierra.png`, `raw/sierra_sierra.png` |
| Hand-labeled data | hand-digitized x, y points of the trajectory: `digitized/charlie_sierra.csv`, `digitized/sierra_sierra.csv` |
| Cleaned trajectory | `charlie_sierra_trajec.parquet`, `sierra_sierra_trajec.parquet` |
| Unifying analysis | `charlie_sierra_unifying_analysis.parquet`, `sierra_sierra_unifying_analysis.parquet` |
| Notebook | `charlie_sierra.ipynb`, `sierra_sierra.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/nautical_search/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

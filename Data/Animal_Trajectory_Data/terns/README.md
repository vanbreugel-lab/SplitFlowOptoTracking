# Foraging terns tracked from a hovering drone in a tidal flow

## Source

Lieber L, Langrock R, Nimmo-Smith WAM (2021). A bird's-eye view on turbulence: seabird foraging associations with evolving surface flow features. *Proceedings of the Royal Society B* 288:20210592.

- DOI: https://doi.org/10.1098/rspb.2021.0592
- Data repository: Dryad: https://doi.org/10.5061/dryad.kh189325b
- Paper: `paper/Lieber_2021_ProcRSocB.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | `raw/terns.csv` (columns ID, x, y, hover, t, speed, ...) holds all tern tracks; `raw/ALL_merged_d200div_d25curl.csv` is a second table of the same tracks with flow features |
| Hand-labeled data | none; the raw data are tracked positions |
| Cleaned trajectory | `terns_Tern1838-1-1_trajec.parquet` |
| Unifying analysis | `terns_Tern1838-1-1_unifying_analysis.parquet` |
| Notebook | `tern_1838-1-1.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/terns/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

## Notes

- The trajectory used is track `Tern1838-1-1`, first 1200 samples, from `raw/terns.csv`.
- `tern_jh.ipynb` explores `ALL_merged_d200div_d25curl.csv`; it writes no data.

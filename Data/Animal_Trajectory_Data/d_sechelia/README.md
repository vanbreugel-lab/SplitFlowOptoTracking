# Flying *Drosophila sechellia* tracking noni odour in a wind tunnel

## Source

Takagi S, Sancer G, Abuin L, Stupski SD, Arguello JR, Prieto-Godino LL, Stern DL, Cruchet S, Alvarez-Ocana R, Wienecke CFR, van Breugel F, Jeanne JM, Auer TO, Benton R (2024). Olfactory sensory neuron population expansions influence projection neuron adaptation and enhance odour tracking. *Nature Communications*.

- DOI: https://doi.org/10.1038/s41467-024-50808-w
- Data repository: none public for the tracking data (the paper provides Source Data and other data on request). The trajectories here come from the wind tunnel tracking file `dsec07_noni.csv`, which is not included.
- Paper: `paper/Takagi_2024_NatCommun.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | single trajectories (object ids 1384 and 3661) cut from the tracking file Files: `raw/d_sechelia_1384.parquet`, `raw/d_sechelia_3661.parquet` |
| Hand-labeled data | none; the raw data are tracked positions |
| Cleaned trajectory | `d_sechelia_1384_trajec.parquet`, `d_sechelia_3661_trajec.parquet` |
| Unifying analysis | `d_sechelia_1384_unifying_analysis.parquet`, `d_sechelia_3661_unifying_analysis.parquet` |
| Notebook | `d_sechelia.ipynb`, `d_sechelia_3661.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/d_sechelia/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

## Notes

- `explore_data.ipynb` is the notebook that cut `raw/d_sechelia_1384.parquet` out of `dsec07_noni.csv`; it cannot be rerun without that file.
- Only `d_sechelia_3661` is in `df_all_species.parquet` (used in the figures).

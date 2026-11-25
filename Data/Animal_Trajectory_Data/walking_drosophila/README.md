# Walking *Drosophila* in a miniature wind tunnel with optogenetic activation of olfactory neurons

## Source

Matheson AMM, Lanz AJ, Medina AM, Licata AM, Currier TA, Syed MH, Nagel KI (2022). A neural circuit for wind-guided olfactory navigation. *Nature Communications* 13:4613.

- DOI: https://doi.org/10.1038/s41467-022-32247-7
- Data repository: Zenodo: https://zenodo.org/records/6863832 (DOI 10.5281/zenodo.6863832). The trajectory here was cut from `MatchedOrco_chrimson.mat` in that record, which is not included here; the record is a 24 GB split archive that has to be downloaded and reassembled to get the file.
- Paper: `paper/Matheson_2022_NatCommun.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | one trajectory (fly 0, stimulus 4) cut from the authors' data file Files: `raw/walking_fly_0_4.parquet` |
| Hand-labeled data | none; the raw data are tracked positions |
| Cleaned trajectory | `walking_fly_0_4_trajec.parquet` |
| Unifying analysis | `walking_fly_0_4_unifying_analysis.parquet` |
| Notebook | `walking_fly_0_4.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/walking_drosophila/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

## Notes

- `walking_fly.ipynb` is the notebook that cut `raw/walking_fly_0_4.parquet` out of `MatchedOrco_chrimson.mat`; it cannot be rerun without that file (download it from the Zenodo record above and put it next to the notebook). The figures only need `raw/walking_fly_0_4.parquet`, which is included.

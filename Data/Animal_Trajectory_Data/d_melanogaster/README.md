# Flying *Drosophila melanogaster* after an optogenetic odour stimulus (this study)

## Source

This study. The three trajectories are single flash-triggered trajectories from the wind tunnel experiments in `Data/Experimental_Fly_Data/` (laminar wind, still air, and unsteady split flow).

- Data repository: same Dryad deposit as the rest of this repository

## Files

| What | Where |
|---|---|
| Raw data | one trajectory per file, cut from the experimental data: `raw/d_mel_<condition>_<date>_<time>_<object id>_<event>.parquet` |
| Hand-labeled data | none; the raw data are tracked positions |
| Cleaned trajectory | `d_mel_laminar_20220927_170437_3462_123_trajec.parquet`, `d_mel_stillair_20220815_162329_5322_124_trajec.parquet`, `d_mel_unsteady_20240110_164820_11858_151_trajec.parquet` |
| Unifying analysis | `d_mel_laminar_20220927_170437_3462_123_unifying_analysis.parquet`, `d_mel_stillair_20220815_162329_5322_124_unifying_analysis.parquet`, `d_mel_unsteady_20240110_164820_11858_151_unifying_analysis.parquet` |
| Notebook | `laminar.ipynb`, `stillair.ipynb`, `unsteady.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/d_melanogaster/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

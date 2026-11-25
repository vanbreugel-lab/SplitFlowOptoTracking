# Wandering albatross (*Diomedea exulans*) approaching prey

## Source

Nevitt GA, Losekoot M, Weimerskirch H (2008). Evidence for olfactory search in wandering albatross, *Diomedea exulans*. *Proceedings of the National Academy of Sciences*.

- DOI: https://doi.org/10.1073/pnas.0709047105
- Data repository: none; the trajectory was digitized by hand from a published figure
- Paper: `paper/Nevitt_2008_PNAS.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | screenshots of Fig. 1C ("Zigzag" approach; called casting here) and Fig. 1D ("Circle" approach); axes are latitude / longitude with a scale bar Files: `raw/albatross_casting.png`, `raw/albatross_circling.png` |
| Hand-labeled data | hand-digitized x, y points of the trajectory: `digitized/albatross_casting_trajec.csv`, `digitized/albatross_circling_trajec.csv` |
| Cleaned trajectory | `albatross_casting_trajec_trajec.parquet`, `albatross_circling_trajec_trajec.parquet` |
| Unifying analysis | `albatross_casting_trajec_unifying_analysis.parquet`, `albatross_circling_trajec_unifying_analysis.parquet` |
| Notebook | `albatross_casting_new.ipynb`, `albatross_circling_new.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/albatross/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

# Mouse following a printed odour trail

## Source

Jayakumar S, Rigolli N, Mathis MW, Vergassola M, Mathis A, Murthy VN (2025). Mice navigate scent trails using predictive policies. bioRxiv preprint.

- DOI: https://doi.org/10.1101/2025.08.27.672631
- Data repository: none listed in the preprint (sample videos: https://github.com/VNMurthyLab/trail_tracking)
- Paper: `paper/Jayakumar_2025_bioRxiv.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | `raw/mouse_straight.png`: screenshot of panel a with two example snout paths (upper: straight trail; lower: tortuous trail; scale bars 100 cm and 10 cm). `raw/from_venki_Trajectory1_0205.csv`: one tracked trial (trail, snout, shoulder and tail-base positions) received from V. N. Murthy; no notebook reads it |
| Hand-labeled data | hand-digitized x, y points of the trajectory: `digitized/mouse_straight.csv`, `digitized/mouse_tortuous.csv` |
| Cleaned trajectory | `mouse_straight_trajec.parquet`, `mouse_tortuous_trajec.parquet` |
| Unifying analysis | `mouse_straight_unifying_analysis.parquet`, `mouse_tortuous_unifying_analysis.parquet` |
| Notebook | `mouse_new.ipynb`, `mouse_tort_new.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/mouse/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

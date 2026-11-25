# Bonnethead shark (*Sphyrna tiburo*) responding to an odour stimulus in a circular pool

## Source

Johnsen PB, Teeter JH (1985). Behavioral responses of bonnethead sharks (*Sphyrna tiburo*) to controlled olfactory stimulation. *Marine Behaviour and Physiology* 11(4):283-291.

- DOI: https://doi.org/10.1080/10236248509387053
- Data repository: none; the trajectory was digitized by hand from a published figure
- Paper: `paper/Johnsen_1985_MarBehavPhysiol.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | screenshots of the four tank drawings: panels A and B (no current; `shark_a`, `shark_b`) and panels C and D (current; `shark_c`, `shark_d`). The red circle marks the tank outline used for scaling Files: `raw/shark_a.png`, `raw/shark_b.png`, `raw/shark_c.png`, `raw/shark_d.png` |
| Hand-labeled data | hand-digitized x, y points of the trajectory: `digitized/shark_a.csv`, `digitized/shark_b.csv`, `digitized/shark_c.csv`, `digitized/shark_d.csv` |
| Cleaned trajectory | `shark_a_trajec.parquet`, `shark_b_trajec.parquet`, `shark_c_trajec.parquet`, `shark_d_trajec.parquet` |
| Unifying analysis | `shark_a_unifying_analysis.parquet`, `shark_b_unifying_analysis.parquet`, `shark_c_unifying_analysis.parquet`, `shark_d_unifying_analysis.parquet` |
| Notebook | `shark_a.ipynb`, `shark_b.ipynb`, `shark_c.ipynb`, `shark_d.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/sharks/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

## Notes

- `shark_alignment.ipynb` plots the fits of shark_a and shark_d side by side; it writes no data.

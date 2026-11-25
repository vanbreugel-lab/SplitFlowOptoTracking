# Reinforcement-learning agent tracking a simulated odour plume

## Source

Singh SH, van Breugel F, Rao RPN, Brunton BW (2023). Emergent behaviour and neural dynamics in artificial agents tracking odour plumes. *Nature Machine Intelligence* 5:58-70.

- DOI: https://doi.org/10.1038/s42256-022-00599-w
- Data repository: figshare (PlumeData): https://figshare.com/articles/dataset/PlumeData/16879539 ; code: https://github.com/BruntonUWBio/plumetracknets (the trajectory here was digitized from the figure, not taken from these)
- Paper: `paper/Singh_2023_NatMachIntell.pdf` (kept locally for reference; not redistributed and not committed)

## Files

| What | Where |
|---|---|
| Raw data | screenshot of panel c, "Sparser (reduced birth and diffusion rate)"; axes in metres Files: `raw/RL_agent_c.png` |
| Hand-labeled data | hand-digitized x, y points of the trajectory: `digitized/rl_agent_c.csv` |
| Cleaned trajectory | `rl_agent_c_trajec.parquet` |
| Unifying analysis | `rl_agent_c_unifying_analysis.parquet` |
| Notebook | `RL_agent_new.ipynb` in `FigureGeneration/Supplemental/supp_fig_14_all_animal_trajectories/animal_data/RL_agents/` of the code repository |

## How the files were made

Each notebook reads the raw or hand-labeled data, scales it to metres and seconds, interpolates it to a regular time base,
and saves the cleaned trajectory as `<name>_trajec.parquet`. It then fits the unifying algorithm to that trajectory and
saves the fit as `<name>_unifying_analysis.parquet`. File locations are listed in `Data/data_paths.yaml`
(keys starting with `animal.`). `../df_all_species.parquet` is the summary table of all animals used by Figs. 5, S14 and S15.

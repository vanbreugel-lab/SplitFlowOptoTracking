# find_augmented_shams

This directory is provided **for reference/provenance only**. The two
notebooks here will **not run** for anyone outside the original lab: they
read raw per-session fly-tracking recordings from an external drive path
hardcoded to the original author's machine
(`/media/caveman/Vorchard1/jaleesa_unsteady/fly_data/...` and
`/media/caveman/Vorchard1/kevin_split_flow_data/wildtypes/...`). That raw,
per-session recording data is not included in this repository and is not
published on Dryad.

- `find_augmented_shams.ipynb` — documents how the six date-stamped
  `*_SHAM_AUGMENTED.parquet` files in this directory, and their merge into
  `flies_splitflow_topon_c1xwt_preprocessed_optotrigger_trimmed_SHAM_AUGMENTED_merged.parquet`,
  were produced from the top-on split-flow C1xWT sessions.
- `find_augmented_shams_wt_data.ipynb` — the analogous notebook for the
  wildtype split-flow dataset.

**You do not need to run either notebook.** Their output —
`flies_splitflow_topon_c1xwt_preprocessed_optotrigger_trimmed_SHAM_AUGMENTED_merged.parquet`
— is already included in `../../raw_data/`, which is what
`analyze_unsteady_exp_top_on_sham_augmented.py` and the figure notebooks
actually read via `config.py`. The parquet files sitting alongside these
notebooks are kept only so the merge step above is reproducible to read.

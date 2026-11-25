# find_augmented_shams

**For reference only.** The two notebooks here read the raw per-night fly-tracking recordings, which are **not
publicly available due to size constraints** (they are not in `Data/` and not on Dryad). The notebooks are kept
because they document how the augmented shams were found. Their locations on the lab server are listed in
`Data/data_paths.yaml` (`fly.raw.splitflow_topon_c1xwt_nights`, `fly.raw.splitflow_topon_wt_nights`).

## Why augmented shams

A sham trajectory for the unsteady (split flow, top on) condition must stay below z = 0.26 m, in the unsteady
part of the flow, for the whole trajectory. Only about 3 of the recorded sham events do. `find_augmented_shams.ipynb`
therefore searches the raw recordings of the same C1xWT nights with looser conditions than the live trigger
allowed, and keeps lights-off trajectories that stay below the split. This gives the sham set used for the
unsteady condition in the figures.

## Files

- `find_augmented_shams.ipynb`: makes one `*_SHAM_AUGMENTED.parquet` file per night from the C1xWT top-on nights
  and merges them into `flies_splitflow_topon_c1xwt_preprocessed_optotrigger_trimmed_SHAM_AUGMENTED_merged.parquet`
  (key `fly.splitflow_topon_c1xwt_sham_augmented`).
- `find_augmented_shams_wt_data.ipynb`: the same search for the wild-type nights. Its output was never used: the
  wild-type sham fits use the recorded wild-type shams.

**You do not need to run either notebook.** The merged C1xWT file is in `Data/Experimental_Fly_Data/`, which is what
`analyze_unsteady_exp_top_on_sham_augmented.py` and the figure notebooks read.

# Unifying simulation animation

1. (Optional) Regenerate `unifying_sim_data.parquet` by running `run_simulation.ipynb`.
2. In `plot_simulation.ipynb`, set `option` (cell 4) to `'original'`, `'smooth'`, or `'saccadic'`, then run all cells from this directory.
   PNG stills are written to `<option>_frames/` (every 2nd frame; change `STRIDE` in the last cell).
   Requires system LaTeX (`text.usetex=True`) and cairo for `cairosvg` (`brew install cairo`).
3. Combine the stills into a movie (example for `saccadic`):

   ```
   ffmpeg -r 30 -pattern_type glob -i 'saccadic_frames/*.png' -c:v libx264 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' out_saccadic.mp4
   ```

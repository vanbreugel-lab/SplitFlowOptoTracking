# RunFigures

Scripts that regenerate the manuscript figures and collect them into `FinalFigures/`.
Run them from the repository root. They activate `.venv` automatically.

## Rerun one figure

```bash
RunFigures/fig3.sh                          # 1. regenerate the figure's SVG
RunFigures/collect_final_figures.sh fig3    # 2. copy it to FinalFigures/{svgs,pdfs,pngs}
```

Step 1 only updates the SVG in `FigureGeneration/`. `FinalFigures/` changes only after step 2.
Script names and figure ids are listed in `figures.yaml` (e.g. `supp_altitude_response`).

## Rerun all figures

```bash
RunFigures/run_all.sh                       # slow; prints a pass/fail summary
RunFigures/collect_final_figures.sh         # all figures
```

## Notes

- `check_code_data.sh` checks, without running anything, that every script and notebook in the repository
  (not only the ones run here) asks for data that is listed in `Data/data_paths.yaml` and is on disk.
  `Data/check_data.sh` checks `Data/` against the same list. `run_all.sh` runs both first and stops if
  either fails (`SKIP_DATA_CHECKS=1` skips them).
- `run_all.sh --dry-run` only checks that the data is in place and that every notebook and script exists.
- Executed notebooks and logs go to `RunFigures/logs/<id>/`. The source notebooks are not modified.
- Fig 5 needs a Gurobi license (`GRB_LICENSE_FILE`), and Fig 1 needs LaTeX.
- Some figures are built from a `*_template.svg`. Make hand edits in the template, not in the output SVG.
- Check fonts: `python RunFigures/audit_fonts.py <figure>.svg --outliers`.

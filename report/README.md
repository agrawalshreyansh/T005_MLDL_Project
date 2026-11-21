# Report (IEEE conference format)

`main.pdf` is the compiled deep-learning report (6 pages, `IEEEtran` conference class).

| File | What it is |
|---|---|
| `main.tex`, `references.bib` | report source and bibliography |
| `tables/*.tex`, `figures/*.pdf` | **generated** by `make_report_assets.py` from the saved results; do not edit by hand |
| `report_facts.json` | numbers quoted in the text (bootstrap results, per-sensor shares), written by the same script |

## Rebuild

```bash
python report/make_report_assets.py        # tables + figures from artifacts/ (~2 min)
cd report && tectonic -X compile main.tex  # -> main.pdf
```

`tectonic` (`brew install tectonic`) downloads the LaTeX packages it needs on first use. With a full TeX
installation instead: `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Overleaf also works:
upload the `report/` folder and compile `main.tex`.

Needs the prediction files and training histories (root README, section 5). The prose numbers in
`main.tex` were checked against the generated tables and `reports/dl_results_summary.md`; re-check them
if you retrain.

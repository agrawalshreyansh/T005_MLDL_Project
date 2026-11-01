# Phase 1 deliverables

| File | What it is |
|---|---|
| `main.tex`, `references.bib` | IEEE conference-format report (class `IEEEtran`). Section order: introduction, related work (literature review), data and EDA, problem formulation and theory, feature engineering, setup, results, discussion, limitations, conclusion. |
| `tables/*.tex` | Result tables, **generated** from `artifacts/metrics/*.csv` by `python report/make_tables.py`. Do not edit by hand; re-run the script after re-running the notebooks. |
| `figures/*.png` | Figures copied from `artifacts/figures/` (the originals are produced by notebooks 01, 02, 05c, 05e and 07a). |
| `slides/phase1_presentation.pptx` | 14-slide presentation with speaker notes. |

## Compiling the report

The report was written without a local LaTeX installation and has **not been compiled yet**. Upload the whole `report/` folder to Overleaf (it provides `IEEEtran`) and compile `main.tex`; or locally:

```bash
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Before submitting: fill in the author block (`\author{...}` is a placeholder), and read the text once against `reports/phase1_results_summary.md`.

## Regenerating the numbers

```bash
python -m pytest tests -q --ignore=tests/test_gat_model.py --ignore=tests/test_dcrnn_model.py
# run notebooks 01-05e and 07a (see the root README), then:
python report/make_tables.py
```

"""Assemble notebooks/OneFileConsistofEverythingInCells.ipynb from the executed notebooks (outputs preserved, nothing re-run).

Usage (from the project root, after executing the individual notebooks):  python tools/merge_notebooks.py
"""
import uuid
from pathlib import Path
import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
ORDER = ["00_environment_and_setup", "01_data_exploration", "02_data_preprocessing", "03_graph_construction", "04_baseline_models",
         "05_xgboost_forecasting", "06_graph_attention_network", "06b_optional_dcrnn", "07_model_evaluation",
         "08_congestion_visualization", "09_optional_sumo_simulation"]

merged = nbf.v4.new_notebook()
merged.metadata = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
merged.cells.append(nbf.v4.new_markdown_cell(
    "# Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models\n\n"
    "**One-file version.** All project notebooks in execution order, each section starting with its own setup cell (so any section can be re-run after a restart). "
    "This file was **assembled from the executed notebooks** (`00` ... `09`): the outputs shown are those of the real runs, nothing was re-run to build it and nothing was edited. "
    "Section 06b (optional DCRNN) is included only if that notebook exists and was executed. Run modes, data requirements and the meaning of every number are explained in `README.md`.\n\n"
    "| Section | Notebook |\n|---|---|\n" + "\n".join(f"| {i} | `{n}` |" for i, n in enumerate(ORDER))))
used = []
for name in ORDER:
    p = NB_DIR / f"{name}.ipynb"
    if not p.is_file():
        print("skipped (missing):", name); continue
    nb = nbf.read(p, as_version=4)
    if not any(c.cell_type == "code" and c.get("outputs") for c in nb.cells):
        print("skipped (not executed):", name); continue
    merged.cells.append(nbf.v4.new_markdown_cell(f"---\n---\n# Section: `{name}`"))
    for c in nb.cells:
        c = nbf.from_dict(c); c["id"] = uuid.uuid4().hex[:8]
        merged.cells.append(c)
    used.append(name)
nbf.validate(merged)
out = NB_DIR / "OneFileConsistofEverythingInCells.ipynb"
nbf.write(merged, out)
print("wrote", out.name, "with sections:", used, f"({len(merged.cells)} cells, {out.stat().st_size / 1e6:.1f} MB)")

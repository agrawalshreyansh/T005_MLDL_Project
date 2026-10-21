"""Structural checks on every notebook: valid nbformat JSON, markdown+code cells, parseable code, no hard-coded absolute paths."""
import ast
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
NOTEBOOKS = [p for p in NOTEBOOKS if not p.name.startswith("_")]


def test_notebooks_exist():
    assert NOTEBOOKS, "no notebooks found"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_well_structured(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    kinds = [c.cell_type for c in nb.cells]
    assert "markdown" in kinds and "code" in kinds
    assert nb.cells[0].cell_type == "markdown", "notebooks start with a markdown objective"
    longest = max((len(c.source.splitlines()) for c in nb.cells if c.cell_type == "code"), default=0)
    assert longest <= 120, f"{path.name}: a single code cell has {longest} lines (split it up)"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse_and_have_no_machine_specific_paths(path):
    nb = nbformat.read(path, as_version=4)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = "\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith(("%", "!")))
        try:
            ast.parse(src)
        except SyntaxError as exc:                    # pragma: no cover - message is the point
            pytest.fail(f"{path.name} cell {i}: {exc}")
        assert "/Users/" not in cell.source and "C:\\" not in cell.source, f"{path.name} cell {i} hard-codes a path"

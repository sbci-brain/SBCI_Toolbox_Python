"""What the package says about installing itself, and what ships with it.

The install advice of every message names the repository, since the package is not on
PyPI; the bundled atlases carry their provenance; CI's minimum-versions job pins exactly the
minimums pyproject.toml declares.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --- the install advice ------------------------------------------------------------------


def _advice(extra):
    return (
        f'pip install "sbci[{extra}] @ git+https://github.com/sbci-brain/SBCI_Toolbox_Python.git"',
        f'pip install -e ".[{extra}]"',
    )


def test_each_missing_extra_names_the_repository_not_pypi(monkeypatch):
    """The advice named the package alone, which pip looks for on PyPI, where it is not."""
    from sbci import _device, plotting, render

    monkeypatch.setitem(sys.modules, "nilearn", None)  # each import then fails, as when missing
    monkeypatch.setitem(sys.modules, "pyvista", None)
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(_device, "_USABLE", None)
    for extra, attempt in (
        ("plotting", plotting._import_nilearn),
        ("render", render._import_pyvista),
        ("gpu", lambda: _device.check("gpu")),
    ):
        with pytest.raises(ImportError) as raised:
            attempt()
        for command in _advice(extra):
            assert command in str(raised.value)
        assert f"'sbci[{extra}]'" not in str(raised.value)


#: ``pip install`` naming the package by its name alone, as PyPI would take it: quoted or
#: not, with or without extras, and not followed by `` @ git+``.
PYPI_FORM = re.compile(
    r"pip\s+install\s+(?:-\S+\s+)*['\"]?"
    + "sb"
    + r"ci(?:\[[^\]]*\])?(?!\[)(?!\s*@\s*git\+)(?![\w-])(?!\.\w)"
)


@pytest.mark.parametrize("folder", ["src", "tests"])
def test_no_install_advice_points_at_pypi(folder):
    """Docstrings, messages and tests: every ``pip install`` of the package names its source."""
    if not (ROOT / folder).is_dir():
        pytest.skip(f"no {folder}/ beside the tests")
    found = []
    for path in sorted((ROOT / folder).rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        # a command broken over two lines of a docstring is one command
        text = " ".join(path.read_text(encoding="utf-8").split())
        found += [f"{path.relative_to(ROOT)}: {m.group(0)}" for m in PYPI_FORM.finditer(text)]
    assert not found

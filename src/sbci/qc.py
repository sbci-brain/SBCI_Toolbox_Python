"""A page to look at one subject's files before they are used: ``sbci qc``.

``sbci qc --sc F --fc F --run-dir RUN --out QCDIR`` writes ``QCDIR/index.html``,
a static page with everything inlined, and prints a JSON verdict:

- every check of ``sbci validate``, for each file;
- the checks that tie the two files together: each holds the modality it is
  given as, both name the same subject, both carry the same vertex areas and
  medial wall, the FC holds some correlation between cortical vertices;
- the measures of :data:`MEASURES`, beside the reference limits when there are
  any;
- figures, when matplotlib is installed: the SC's strength over the surface and
  its Desikan matrix, the FC's distribution, its mean over the surface and its
  Schaefer-200 matrix, and with both files their coupling;
- what the pipeline run recorded, from ``RUN``: ``config/versions.json``,
  ``sbci_local.json``, ``local_adapter_manifest.json``, ``logs/convert.json``
  and every non-empty ``logs/*.err``.

The verdict is ``fail`` when a check fails, ``flag`` when every check passes
but a measure lies outside its reference limits, and ``pass`` otherwise; the
exit status is 0 for ``pass`` and 1 for the others. The limits come from a
reference population, the released subjects, never from a pilot: they are
built by ``tools/build_qc_reference.py`` and read from ``--limits``,
``$SBCI_QC_LIMITS`` or the package's own file, in that order. None is shipped
yet, and without one the verdict rests on the checks alone, which the page
says.

The page is written first as a placeholder, then once everything is known, so
there is a page whatever fails part way.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
import numbers
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, spec
from .errors import SbciError

MEASURES_VERSION = 1
"""Bumped whenever a measure of :data:`MEASURES` changes; limits built for another are not used."""

MEASURES: dict[str, str] = {
    "sc_streamline_count": "streamlines the SC was smoothed from, as its metadata records them",
    "sc_interhemispheric_share": "share of the SC's area-weighted mass between the hemispheres",
    "sc_max_vertex_share": "largest share of the SC's area-weighted mass at a single vertex",
    "fc_frames": "frames the FC was computed from, as its metadata records them",
    "fc_mean": "mean correlation over the pairs of cortical vertices",
    "fc_sd": "standard deviation of the correlations over the pairs of cortical vertices",
    "coupling_median": "median over cortex of the SC-FC coupling, global scope (a cosine)",
}
"""What ``sbci qc`` measures, and ``tools/build_qc_reference.py`` takes limits for, alike."""

FILE_FIELDS = (
    "streamline_count",
    "streamline_weighting",
    "kernel",
    "bandwidth",
    "endpoint_source",
    "fc_frames",
    "fc_runs",
    "fc_nuisance_model",
    "normalization",
    "conversion",
    "entry_point",
    "pipeline_version",
    "container_version",
    "provenance",
)
"""The metadata the page shows of each file; ``conversion``, the shares ``sbci convert`` removed."""

LIMITS_FORMAT = "sbci-qc-limits"
LIMITS_ENVIRONMENT = "SBCI_QC_LIMITS"
LIMITS_RESOURCE = "qc_limits.json"
"""The package's own limits, in ``sbci/data``, once limits from the released subjects are built."""

_PLOTTING = "pip install 'sbci[plotting]'"


class QcUsageError(SbciError, ValueError):
    """The command line names something that is not there: exit status 2."""


# --- measures ---------------------------------------------------------------------------


def cortex_pairs(mask: np.ndarray) -> np.ndarray:
    """Which entries of the condensed upper triangle join two cortical vertices.

    Examples
    --------
    >>> cortex_pairs(np.array([True, True, False])).tolist()
    [True, False, False]
    """
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    return np.concatenate([mask[i + 1 :] & mask[i] for i in range(n - 1)] or [np.zeros(0, bool)])


def _count(value) -> float | None:
    """A positive count from the metadata, or ``None`` when it is not recorded."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


@dataclass
class Survey:
    """The measures of one subject's files, and the arrays their figures draw."""

    values: dict[str, float | None] = field(default_factory=dict)
    sc_strength: np.ndarray | None = None
    fc_histogram: tuple[np.ndarray, np.ndarray] | None = None
    fc_mean_map: np.ndarray | None = None
    coupling: np.ndarray | None = None


def survey(sc=None, fc=None, maps: bool = True) -> Survey:
    """Measure the files: :data:`MEASURES`, and with ``maps`` the arrays the page draws.

    ``sc`` and ``fc`` are :class:`~sbci.ContinuousConnectome` objects, either
    may be ``None``. A measure that cannot be taken is ``None``.
    """
    result = Survey()
    values = result.values
    if sc is not None:
        values["sc_streamline_count"] = _count(sc.metadata.get("streamline_count"))
        dense = sc.dense(np.float64)
        area = np.asarray(sc.area, dtype=np.float64)
        strength = dense @ area
        mass = area * strength
        total = float(mass.sum())
        half = sc.n_vertices // 2
        between = 2.0 * float(area[:half] @ dense[:half, half:] @ area[half:])
        del dense
        values["sc_interhemispheric_share"] = between / total if total > 0 else None
        values["sc_max_vertex_share"] = float(mass.max()) / total if total > 0 else None
        if maps:
            result.sc_strength = np.where(sc.mask, strength, np.nan)
    if fc is not None:
        values["fc_frames"] = _count(fc.metadata.get("fc_frames"))
        mask = np.asarray(fc.mask, dtype=bool)
        r = np.asarray(fc.data)[cortex_pairs(mask)].astype(np.float64)
        values["fc_mean"] = float(r.mean()) if r.size else None
        values["fc_sd"] = float(r.std()) if r.size else None
        if maps:
            result.fc_histogram = np.histogram(r, bins=np.linspace(-1.0, 1.0, 81))
            dense = fc.dense(np.float32)
            dense[:, ~mask] = 0.0
            others = max(int(mask.sum()) - 1, 1)
            mean = dense.sum(axis=1, dtype=np.float64) / others
            del dense
            result.fc_mean_map = np.where(mask, mean, np.nan)
        del r
    if sc is not None and fc is not None:
        from .coupling import structure_function_coupling

        coupling = np.asarray(structure_function_coupling(sc, fc, scope="global"), float)
        cortex = np.asarray(sc.mask, bool) & np.asarray(fc.mask, bool) & np.isfinite(coupling)
        values["coupling_median"] = float(np.median(coupling[cortex])) if cortex.any() else None
        if maps:
            result.coupling = np.where(cortex, coupling, np.nan)
    return result


def measures(sc=None, fc=None) -> dict[str, float | None]:
    """The measures of :data:`MEASURES` for one subject, as ``sbci qc`` and the limits take them."""
    return survey(sc, fc, maps=False).values


# --- reference limits -------------------------------------------------------------------


def load_limits(path: str | Path | None = None) -> tuple[dict | None, str]:
    """``(limits, where)``: the reference limits to gate on, or ``None`` and why there are none.

    Looked for in ``path``, then ``$SBCI_QC_LIMITS``, then the package's own
    file. A file that is there but cannot be used raises :class:`ValueError`;
    a ``path`` that is not there raises :class:`QcUsageError`.
    """
    if path is not None:
        if not Path(path).is_file():
            raise QcUsageError(f"--limits {path} is not a file")
        return _read_limits(Path(path)), str(path)
    named = os.environ.get(LIMITS_ENVIRONMENT)
    if named:
        if not Path(named).is_file():
            raise ValueError(f"{LIMITS_ENVIRONMENT} names {named}, which is not a file")
        return _read_limits(Path(named)), named
    from importlib import resources

    bundled = resources.files("sbci.data") / LIMITS_RESOURCE
    if bundled.is_file():
        return _read_limits(Path(str(bundled))), f"the package's {LIMITS_RESOURCE}"
    return None, "no reference limits installed"


def _read_limits(path: Path) -> dict:
    """A limits file, refused unless it is one this version of the measures can use."""
    try:
        limits = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} cannot be read as JSON: {exc}") from exc
    if not isinstance(limits, dict) or limits.get("format") != LIMITS_FORMAT:
        raise ValueError(f"{path} is not a {LIMITS_FORMAT} file")
    if limits.get("measures_version") != MEASURES_VERSION:
        raise ValueError(
            f"{path} holds limits for measures version {limits.get('measures_version')!r}, and "
            f"this sbci measures version {MEASURES_VERSION}: rebuild it"
        )
    table = limits.get("limits")
    if not isinstance(table, dict):
        raise ValueError(f"{path} has no table of limits")
    for name, row in table.items():
        try:
            low, high = float(row["low"]), float(row["high"])
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"{path}: the limits of {name} are not two numbers") from exc
        if name not in MEASURES or not low <= high:
            raise ValueError(f"{path}: {name} is not a measure with low <= high")
    return limits


def gate(values: dict[str, float | None], limits: dict | None) -> list[dict[str, Any]]:
    """Each measure beside its limits: ``within`` is ``None`` where there is nothing to compare."""
    table = (limits or {}).get("limits", {})
    rows = []
    for name, description in MEASURES.items():
        if name not in values:
            continue
        value = values[name]
        limit = table.get(name)
        within = None
        if limit is not None and value is not None:
            within = bool(float(limit["low"]) <= value <= float(limit["high"]))
        rows.append(
            {
                "name": name,
                "description": description,
                "value": value,
                "low": None if limit is None else float(limit["low"]),
                "high": None if limit is None else float(limit["high"]),
                "within": within,
            }
        )
    return rows


# --- the run's records ------------------------------------------------------------------


def _json_file(path: Path) -> tuple[Any, str | None]:
    """A JSON file's contents and ``None``, or ``None`` and what is wrong with it."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"cannot be read: {exc}"
    if not text.strip():
        return None, "is empty"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"is not JSON: {exc}"


def _tail(path: Path, lines: int = 40, limit: int = 1 << 16) -> str:
    """The last ``lines`` lines of a text file, from at most its last ``limit`` bytes."""
    with path.open("rb") as handle:
        handle.seek(max(path.stat().st_size - limit, 0))
        text = handle.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def run_information(run: str | Path | None) -> dict[str, Any]:
    """What a pipeline run folder recorded, as the page shows it; nothing here raises."""
    if run is None:
        return {}
    run = Path(run)
    info: dict[str, Any] = {"folder": str(run), "files": [], "logs": []}
    if not run.is_dir():
        info["problem"] = f"{run} is not a folder"
        return info
    for relative in (
        "config/versions.json",
        "config/sbci_local.json",
        "sbci_input/sbci_local.json",
        "local_adapter_manifest.json",
        "sbci_output/local_adapter_manifest.json",
        "logs/convert.json",
    ):
        path = run / relative
        if path.is_file():
            contents, problem = _json_file(path)
            info["files"].append({"file": relative, "contents": contents, "problem": problem})
    logs = run / "logs"
    if logs.is_dir():
        for path in sorted(logs.glob("*.err")):
            # qc.err is this command's own stderr, being written as it runs
            if path.name == "qc.err" or not path.is_file() or path.stat().st_size == 0:
                continue
            try:
                tail = _tail(path)
            except OSError as exc:
                tail = f"(cannot be read: {exc})"
            info["logs"].append(
                {"file": f"logs/{path.name}", "bytes": path.stat().st_size, "tail": tail}
            )
    return info


def _recorded(info: dict[str, Any], relative: str) -> dict[str, Any] | None:
    """One of the run's files as :func:`run_information` read it."""
    return next((f for f in info.get("files", []) if f["file"] == relative), None)


# --- the verdict ------------------------------------------------------------------------


@dataclass
class Result:
    """Everything the page and the JSON verdict say."""

    out: Path
    sc: str | None = None
    fc: str | None = None
    subject: str | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    measures: list[dict[str, Any]] = field(default_factory=list)
    limits: dict | None = None
    limits_from: str = "no reference limits installed"
    figures: list[dict[str, Any]] = field(default_factory=list)
    figure_note: str = ""
    run: dict[str, Any] = field(default_factory=dict)
    file_versions: dict[str, Any] = field(default_factory=dict)
    file_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    started: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S %z"))

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        """Record one check."""
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    @property
    def status(self) -> str:
        """``fail`` on a failed check, ``flag`` on a measure outside its limits, else ``pass``."""
        if not self.checks or not all(c["passed"] for c in self.checks):
            return "fail"
        if any(m["within"] is False for m in self.measures):
            return "flag"
        return "pass"

    @property
    def page(self) -> Path:
        """Where the page is written."""
        return self.out / "index.html"

    def versions(self) -> dict[str, Any]:
        """The versions the verdict names: this package's, the run's and the files'."""
        versions: dict[str, Any] = {"sbci": __version__, "spec_version": spec.SPEC_VERSION}
        recorded = _recorded(self.run, "config/versions.json")
        if recorded and isinstance(recorded["contents"], dict):
            versions["run"] = recorded["contents"]
        if self.file_versions:
            versions["files"] = self.file_versions
        return versions

    def as_dict(self) -> dict[str, Any]:
        """The JSON verdict."""
        return {
            "command": "sbci qc",
            "status": self.status,
            "subject": self.subject,
            "checks": self.checks,
            "measures": self.measures,
            "limits": self.limits_from,
            "page": str(self.page.resolve()),
            "versions": self.versions(),
        }


def qc(
    sc: str | Path | None = None,
    fc: str | Path | None = None,
    run_dir: str | Path | None = None,
    out: str | Path = "qc",
    limits: str | Path | None = None,
    figures: bool = True,
) -> Result:
    """Check one subject's files and write ``out/index.html``; see the module notes.

    Raises :class:`QcUsageError` when ``limits`` names no file, and
    :class:`OSError` when ``out`` cannot be made; anything else is a check on
    the page.
    """
    result = Result(
        out=Path(out), sc=None if sc is None else str(sc), fc=None if fc is None else str(fc)
    )
    try:
        result.limits, result.limits_from = load_limits(limits)
    except QcUsageError:
        raise
    except ValueError as exc:
        result.limits_from = f"not used: {exc}"
        result.check("reference limits", False, str(exc))
    result.out.mkdir(parents=True, exist_ok=True)
    _write_page(result.page, _placeholder(result))
    try:
        _assess(result, run_dir, figures)
    except Exception as exc:  # noqa: BLE001 - the page has to say what stopped the checks
        result.check("qc", False, f"the checks stopped: {type(exc).__name__}: {exc}")
    finally:
        try:
            text = render(result)
        except Exception as exc:  # noqa: BLE001 - a page has to be written whatever fails
            text = _placeholder(result, f"the page could not be drawn: {type(exc).__name__}: {exc}")
        _write_page(result.page, text)
    return result


def _assess(result: Result, run_dir, figures: bool) -> None:
    """Every check, measure and figure, recorded on ``result`` as each is known."""
    from .cohort import parse_name
    from .connectome import ContinuousConnectome
    from .validate import validate_file

    result.run = run_information(run_dir)
    if not result.sc and not result.fc:
        result.check("files", False, "no file was given: pass --sc, --fc or both")
    loaded = {}
    for modality, path in (("sc", result.sc), ("fc", result.fc)):
        if path is None:
            continue
        checks = validate_file(path)
        for check in checks:
            result.check(f"{modality}: validate {check.name}", check.passed, check.detail)
        if not checks[0].passed:  # unreadable: there is nothing to load
            continue
        try:
            connectome = ContinuousConnectome.load(path)
        except (OSError, SbciError) as exc:
            result.check(f"{modality}: load", False, str(exc))
            continue
        result.check(
            f"{modality}: modality",
            connectome.modality == modality,
            f"{Path(path).name} holds {connectome.modality.upper()}",
        )
        result.file_versions[modality] = {
            key: connectome.metadata.get(key)
            for key in ("pipeline_version", "container_version", "spec_version")
        }
        fields = connectome.metadata.fields
        result.file_info[modality] = {key: fields[key] for key in FILE_FIELDS if key in fields}
        if connectome.endpoints is not None:
            result.file_info[modality]["endpoints stored"] = connectome.endpoints.n_streamlines
        loaded[modality] = connectome

    names = {m: parse_name(Path(p).name) for m, p in (("sc", result.sc), ("fc", result.fc)) if p}
    known = {m: n for m, n in names.items() if n is not None}
    if known:
        subject, session, _ = next(iter(known.values()))
        result.subject = f"{subject}_ses-{session}" if session else subject
    if len(known) == 2:

        def label(name) -> str:
            return name[0] + (f" session {name[1]}" if name[1] else "")

        result.check(
            "same subject",
            known["sc"][:2] == known["fc"][:2],
            f"the SC file names {label(known['sc'])}, the FC file {label(known['fc'])}",
        )
    sc, fc = loaded.get("sc"), loaded.get("fc")
    if sc is not None and sc.modality != "sc":
        sc = None
    if fc is not None and fc.modality != "fc":
        fc = None
    if sc is not None and fc is not None:
        same = np.array_equal(sc.area, fc.area) and np.array_equal(sc.mask, fc.mask)
        result.check(
            "same grid",
            same,
            "the two files carry the same vertex areas and medial wall"
            if same
            else "the two files disagree on the vertex areas or the medial wall",
        )
    if fc is not None:
        pairs = cortex_pairs(fc.mask)
        held = int(np.count_nonzero(np.asarray(fc.data)[pairs]))
        result.check(
            "fc: values",
            held > 0,
            f"{held:,} of {int(pairs.sum()):,} pairs of cortical vertices hold a correlation",
        )
    _convert_record(result)

    measured = survey(sc, fc, maps=figures)
    result.measures = gate(measured.values, result.limits)
    if figures:
        _figures(result, sc, fc, measured)
    else:
        result.figure_note = "figures were not asked for"


def _convert_record(result: Result) -> None:
    """The conversion's own verdict, from the run's ``logs/convert.json``."""
    record = _recorded(result.run, "logs/convert.json")
    if record is None:
        return
    contents = record["contents"]
    if record["problem"] or not isinstance(contents, dict):
        result.check(
            "convert",
            False,
            f"logs/convert.json {record['problem'] or 'is not a report'}: sbci convert stopped "
            "before reporting; see logs/convert.err",
        )
        return
    errors = contents.get("errors") or []
    result.check(
        "convert",
        contents.get("status") == "written" and not errors,
        "; ".join(str(e) for e in errors)
        or f"sbci convert wrote {', '.join(contents.get('written') or {})}",
    )


# --- figures ----------------------------------------------------------------------------


def _figures(result: Result, sc, fc, measured: Survey) -> None:
    """Draw what the plotting libraries allow; a figure that fails says so on the page."""
    try:
        from matplotlib.figure import Figure
    except ImportError:
        result.figure_note = (
            f"matplotlib is not installed, so the page has no figures ({_PLOTTING})"
        )
        return
    try:
        import nilearn  # noqa: F401 - plot_surface draws through it
    except ImportError:
        surfaces = False
        result.figure_note = (
            f"nilearn is not installed, so the surface maps are left out ({_PLOTTING})"
        )
    else:
        surfaces = True

    def draw(title: str, caption: str, make) -> None:
        try:
            png = make()
        except Exception as exc:  # noqa: BLE001 - one figure failing leaves the others
            result.figures.append(
                {"title": title, "caption": f"not drawn: {type(exc).__name__}: {exc}"}
            )
        else:
            result.figures.append({"title": title, "caption": caption, "png": png})

    def surface(values, connectome, cmap, label):
        import matplotlib.pyplot as plt

        from .plotting import plot_surface

        figure = plot_surface(values, connectome=connectome, cmap=cmap, title=label, mesh="ico4")
        try:
            return _png(figure)
        finally:
            plt.close(figure)

    def matrix(values, cmap, label, symmetric):
        figure = Figure(figsize=(5.2, 4.4))
        axes = figure.subplots()
        shown = np.asarray(values, dtype=np.float64)
        if symmetric:
            bound = float(np.nanmax(np.abs(shown))) or 1.0
            image = axes.imshow(shown, cmap=cmap, vmin=-bound, vmax=bound, interpolation="nearest")
        else:
            image = axes.imshow(shown, cmap=cmap, interpolation="nearest")
        axes.set_xticks([])
        axes.set_yticks([])
        figure.colorbar(image, ax=axes, shrink=0.85, label=label)
        return _png(figure)

    if sc is not None:
        if surfaces:
            draw(
                "SC strength",
                "Each vertex's connectivity summed over the surface, area-weighted; the medial "
                "wall is grey.",
                lambda: surface(measured.sc_strength, sc, "Blues", "SC strength"),
            )
        draw(
            "SC, Desikan regions",
            "The SC's mass between the 68 Desikan regions, on a log scale.",
            lambda: matrix(_log_mass(sc.to_atlas("Desikan")), "Blues", "log10 mass", False),
        )
    if fc is not None:
        draw(
            "FC distribution",
            "The correlations between pairs of cortical vertices.",
            lambda: _histogram(Figure, *measured.fc_histogram),
        )
        if surfaces:
            draw(
                "FC mean",
                "Each vertex's mean correlation with the rest of cortex.",
                lambda: surface(measured.fc_mean_map, fc, "RdBu_r", "mean FC"),
            )
        draw(
            "FC, Schaefer-200 regions",
            "The mean correlation between the 200 Schaefer regions (Fisher z average).",
            lambda: matrix(fc.to_atlas("Schaefer200"), "RdBu_r", "correlation", True),
        )
    if sc is not None and fc is not None and surfaces:
        median = dict((m["name"], m["value"]) for m in result.measures).get("coupling_median")
        draw(
            "SC-FC coupling",
            "Each vertex's coupling: the cosine between its SC and FC profiles"
            + (f"; median {median:.3f}." if median is not None else "."),
            lambda: surface(measured.coupling, sc, "RdBu_r", "coupling"),
        )


def _log_mass(matrix: np.ndarray) -> np.ndarray:
    """``log10`` of a mass matrix, an empty entry shown as missing."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.log10(np.where(matrix > 0, matrix, np.nan))


def _histogram(Figure, counts: np.ndarray, edges: np.ndarray) -> bytes:
    """The FC distribution as bars on a correlation axis."""
    figure = Figure(figsize=(5.2, 3.2))
    axes = figure.subplots()
    axes.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color="#3b6ea8",
        edgecolor="white",
        linewidth=0.5,
    )
    axes.set_xlim(-1, 1)
    axes.set_xlabel("correlation")
    axes.set_ylabel("pairs of cortical vertices")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    return _png(figure)


def _png(figure) -> bytes:
    """A figure as PNG bytes."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    return buffer.getvalue()


# --- the page ---------------------------------------------------------------------------


_STYLE = """
:root { color-scheme: light; }
body { font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1d2329;
  background: #ffffff; max-width: 1080px; margin: 24px auto; padding: 0 16px; }
h1 { font-size: 1.5em; margin-bottom: 4px; } h2 { font-size: 1.2em; margin-top: 32px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th, td { border-bottom: 1px solid #dde2e6; padding: 6px 8px; text-align: left;
  vertical-align: top; }
th { color: #4a545e; font-weight: 600; }
td.mark { white-space: nowrap; font-weight: 600; }
.pass { color: #1e6b3a; } .flag { color: #8a5a00; } .fail { color: #a3262a; }
.verdict { display: inline-block; padding: 4px 12px; border-radius: 6px; font-weight: 700;
  border: 2px solid currentColor; }
.note { color: #4a545e; }
figure { display: inline-block; margin: 8px 16px 16px 0; vertical-align: top; max-width: 100%; }
figure img { max-width: 100%; height: auto; border: 1px solid #dde2e6; }
figcaption { color: #4a545e; font-size: 0.9em; max-width: 520px; }
pre { background: #f5f7f8; padding: 10px; overflow-x: auto; font-size: 0.85em; }
code { overflow-wrap: anywhere; }
"""


def _e(value: Any) -> str:
    """Escaped text."""
    return html.escape(str(value), quote=True)


def _number(value: float | None) -> str:
    """A measure as the page prints it."""
    if value is None:
        return "not measured"
    if float(value).is_integer() and abs(value) >= 1:
        return f"{int(value):,}"
    return f"{value:.4g}"


def _placeholder(result: Result, problem: str = "") -> str:
    """The page written before anything is known, and if the full page cannot be drawn."""
    reason = (
        problem or "sbci qc started and has not finished; if this page stays, it stopped part way."
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QC</title>'
        f"<style>{_STYLE}</style></head><body><h1>Quality control</h1>"
        f'<p class="verdict fail">NOT FINISHED</p><p>{_e(reason)}</p>'
        f'<p class="note">sbci {_e(__version__)}, {_e(result.started)}</p></body></html>'
    )


def render(result: Result) -> str:
    """The page, as one self-contained HTML document."""
    status = result.status
    words = {
        "pass": "PASS: every check passed",
        "flag": "FLAG: every check passed, and a measure lies outside its reference limits",
        "fail": "FAIL: a check failed",
    }
    files = (
        "".join(
            f"<li>{m.upper()}: <code>{_e(p)}</code></li>"
            for m, p in (("sc", result.sc), ("fc", result.fc))
            if p
        )
        or "<li>no file was given</li>"
    )
    if result.limits is None:
        limits = (
            f"<p><b>Reference limits:</b> {_e(result.limits_from)}. The verdict rests on the "
            "checks alone; the measures are shown without a range.</p>"
        )
    else:
        provenance = result.limits.get("provenance", {})
        limits = (
            f"<p><b>Reference limits:</b> from {_e(result.limits_from)}: "
            f"{_e(provenance.get('source', 'source not recorded'))}, "
            f"{_e(provenance.get('n_subjects', '?'))} subjects, quantiles "
            f"{_e(result.limits.get('quantiles'))}, built {_e(provenance.get('built', '?'))} by "
            f"sbci {_e(provenance.get('sbci_version', '?'))}.</p>"
        )
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>QC {_e(result.subject or '')}</title><style>{_STYLE}</style></head><body>",
        f"<h1>Quality control{': ' + _e(result.subject) if result.subject else ''}</h1>",
        f'<p class="verdict {status}">{_e(words[status])}</p>',
        f"<ul>{files}</ul>",
        limits,
        _files_section(result.file_info),
        "<h2>Checks</h2>",
        _checks_table(result.checks),
        "<h2>Measures</h2>",
        _measures_table(result.measures, result.limits is not None),
        "<h2>Figures</h2>",
    ]
    if result.figure_note:
        parts.append(f'<p class="note">{_e(result.figure_note)}</p>')
    for item in result.figures:
        if "png" in item:
            data = base64.b64encode(item["png"]).decode("ascii")
            parts.append(
                f'<figure><img src="data:image/png;base64,{data}" alt="{_e(item["title"])}">'
                f"<figcaption><b>{_e(item['title'])}.</b> {_e(item['caption'])}"
                "</figcaption></figure>"
            )
        else:
            parts.append(f'<p class="note"><b>{_e(item["title"])}</b>: {_e(item["caption"])}</p>')
    if not result.figures and not result.figure_note:
        parts.append('<p class="note">No file could be drawn.</p>')
    parts.append(_run_section(result.run))
    parts.append(
        f'<p class="note">Written by sbci {_e(__version__)} (format {_e(spec.SPEC_VERSION)}), '
        f"started {_e(result.started)}.</p></body></html>"
    )
    return "\n".join(parts)


def _files_section(info: dict[str, dict[str, Any]]) -> str:
    """What each file says of itself: how it was made, and what its conversion removed."""
    if not info:
        return ""
    parts = ["<h2>Files</h2>"]
    for modality, fields in info.items():
        rows = "".join(
            f"<tr><th>{_e(key)}</th><td><code>{_e(_text(value))}</code></td></tr>"
            for key, value in fields.items()
        )
        parts.append(f"<h3>{_e(modality.upper())}</h3><table>{rows}</table>")
    return "".join(parts)


def _checks_table(checks: list[dict[str, Any]]) -> str:
    rows = "".join(
        f'<tr><td class="mark {"pass" if c["passed"] else "fail"}">'
        f"{'PASS' if c['passed'] else 'FAIL'}</td><td>{_e(c['name'])}</td>"
        f"<td>{_e(c['detail'])}</td></tr>"
        for c in checks
    )
    return f"<table><tr><th></th><th>check</th><th>detail</th></tr>{rows}</table>"


def _measures_table(rows: list[dict[str, Any]], with_limits: bool) -> str:
    if not rows:
        return '<p class="note">No file could be measured.</p>'
    body = []
    for row in rows:
        if row["within"] is None:
            mark = '<td class="mark note">-</td>'
        elif row["within"]:
            mark = '<td class="mark pass">WITHIN</td>'
        else:
            mark = '<td class="mark flag">OUTSIDE</td>'
        span = (
            f"{_number(row['low'])} to {_number(row['high'])}"
            if row["low"] is not None
            else "no limits"
        )
        body.append(
            f'<tr>{mark}<td><code>{_e(row["name"])}</code><br><span class="note">'
            f"{_e(row['description'])}</span></td><td>{_number(row['value'])}</td>"
            f"<td>{_e(span) if with_limits else 'no limits installed'}</td></tr>"
        )
    return (
        "<table><tr><th></th><th>measure</th><th>value</th><th>reference range</th></tr>"
        + "".join(body)
        + "</table>"
    )


def _run_section(info: dict[str, Any]) -> str:
    """The run's records: versions, configuration, manifests, the conversion and its logs."""
    if not info:
        return '<h2>The run</h2><p class="note">No --run-dir was given.</p>'
    parts = [f"<h2>The run</h2><p>Folder <code>{_e(info['folder'])}</code></p>"]
    if info.get("problem"):
        parts.append(f'<p class="fail">{_e(info["problem"])}</p>')
        return "".join(parts)
    if not info["files"] and not info["logs"]:
        parts.append(
            '<p class="note">The folder holds none of the records a pipeline run writes.</p>'
        )
    for item in info["files"]:
        parts.append(f"<h3><code>{_e(item['file'])}</code></h3>")
        if item["problem"]:
            parts.append(f'<p class="fail">{_e(item["file"])} {_e(item["problem"])}</p>')
            continue
        contents = item["contents"]
        summary = _summary(item["file"], contents)
        if summary:
            parts.append(summary)
        shown = json.dumps(contents, indent=2, sort_keys=False, default=str)
        parts.append(f"<details><summary>all of it</summary><pre>{_e(shown)}</pre></details>")
    for log in info["logs"]:
        parts.append(
            f"<h3><code>{_e(log['file'])}</code> ({log['bytes']:,} bytes; the last lines)</h3>"
            f"<pre>{_e(log['tail'])}</pre>"
        )
    return "".join(parts)


def _summary(name: str, contents: Any) -> str:
    """The lines of a record worth reading first, as a table."""
    if not isinstance(contents, dict):
        return ""
    if name.endswith("versions.json"):
        rows = contents.items()
    elif name.endswith("sbci_local.json"):
        rows = [
            (k, contents.get(k))
            for k in ("mode", "approved", "parameters", "workflow")
            if k in contents
        ]
    elif name.endswith("local_adapter_manifest.json"):
        execution = (contents.get("workflow") or {}).get("execution") or {}
        rows = [
            ("status", contents.get("status")),
            ("stage", execution.get("stage")),
            ("state", execution.get("state")),
        ] + [
            ("issue", f"{i.get('code')}: {i.get('message', '')}")
            for i in contents.get("issues", [])
            if isinstance(i, dict)
        ]
    elif name.endswith("convert.json"):
        rows = [
            (k, contents.get(k))
            for k in ("status", "written", "skipped", "removed", "warnings", "errors")
        ]
    else:
        return ""
    cells = "".join(
        f"<tr><th>{_e(k)}</th><td><code>{_e(_text(v))}</code></td></tr>" for k, v in rows
    )
    return f"<table>{cells}</table>"


def _text(value: Any) -> str:
    """A record's value as one line: text as it is, anything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _write_page(path: Path, text: str) -> None:
    """Write the page under a temporary name beside it and move it into place."""
    from .export import _temporary

    temporary = _temporary(path, f".{path.name}.", ".partial")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# --- the command ------------------------------------------------------------------------


def command(args) -> int:
    """``sbci qc``: print the verdict as JSON; 0 for pass, 1 for flag or fail, 2 for usage."""
    try:
        result = qc(sc=args.sc, fc=args.fc, run_dir=args.run_dir, out=args.out, limits=args.limits)
    except QcUsageError as exc:
        print(f"sbci qc: error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"sbci qc: error: {args.out} cannot hold the page: {exc}", file=sys.stderr)
        print(
            json.dumps({"command": "sbci qc", "status": "fail", "page": None, "errors": [str(exc)]})
        )
        return 1
    print(json.dumps(result.as_dict(), indent=2, default=str))
    if result.status != "pass":
        failed = [c["name"] for c in result.checks if not c["passed"]]
        flagged = [m["name"] for m in result.measures if m["within"] is False]
        print(
            f"sbci qc: {result.status}: "
            + "; ".join(filter(None, [", ".join(failed), ", ".join(flagged)]))
            + f"; see {result.page}",
            file=sys.stderr,
        )
    return 0 if result.status == "pass" else 1

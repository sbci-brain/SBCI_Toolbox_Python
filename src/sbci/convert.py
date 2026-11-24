"""The processing pipeline's output as the package's files: ``sbci convert``.

The SBCI processing pipeline leaves one subject's connectivity in its
``sbci_connectome`` folder, as MATLAB and NumPy files:

``smoothed_sc_avg_<bandwidth>_<grid>.mat``
    ``sc``, the structural connectome ``c3_main`` smoothed with the spherical
    heat kernel, as ``np.triu`` writes it: the upper triangle with the diagonal,
    the lower triangle empty, in ``c3_main``'s own units. The name carries the
    bandwidth and the grid.
``fc_avg_<grid>.mat``
    ``fc``, the vertex-by-vertex correlations, in the same layout with ones on
    the diagonal.
``snapped_fibers.npz`` with ``lh_sphere_reg_lps.vtk`` and ``rh_sphere_reg_lps.vtk``
    the streamlines' ends on the subject's surfaces and its registered spheres,
    the branch the smoothed connectome was built from
    (:meth:`sbci.smoothing.Endpoints.from_snapped`).
    ``mesh_intersections_<grid>.mat`` is the pipeline's other, unsnapped
    branch, which does not re-smooth to the stored connectome
    (:meth:`sbci.smoothing.Endpoints.from_matlab`).
``fc_ts.npz``
    the vertex time series the correlations came from: its length is the
    number of frames.

:func:`convert` reads them and writes ``sub-<id>_sc.h5`` and ``sub-<id>_fc.h5``.
The lower triangle has to be empty and every value finite. The diagonal is
dropped and its share of the area-weighted mass reported; an SC with more than
:data:`DIAGONAL_LIMIT` of its mass there is refused, since ``c3_main`` writes
none. The medial wall is zeroed and its share reported. SC is scaled to unit
mass with the bundled vertex areas, which are compared with the run's own
``mapping_avg_ico4.npz`` where it can be found; FC stays correlations. Each
file is written under a temporary name, checked by
:func:`sbci.validate.validate_file`, and renamed only once it passes, so the
name the package reads never holds a file that failed a check.

:func:`clean` is the conversion itself, shared with ``tools/import_legacy.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, spec
from .errors import SbciError

DIAGONAL_LIMIT = 1e-6
"""The largest share of an SC's area-weighted mass its diagonal may hold.

``c3_main`` writes no self-connection: every one of the 5124 diagonal entries
of the pipeline's ``smoothed_sc_avg_0.005_ico4.mat`` is exactly zero
(SPEC_QUESTIONS.md item 2). Mass there means the matrix came from something
else, and dropping it would change the connectome without a word."""

GRID = "ico4"
"""The one grid the package reads, as the pipeline names it in its file names."""

SC_NAME = re.compile(r"^smoothed_sc_avg_(?P<bandwidth>[^_]+)_(?P<grid>[^_]+)\.mat$")
"""``smoothed_sc_avg_<bandwidth>_<grid>.mat``, as the pipeline's structural step names it."""

FC_NAME = re.compile(r"^fc_avg_(?P<grid>[^_]+)\.mat$")
"""``fc_avg_<grid>.mat``; ``fc_avg_<grid>_ts.mat``, the mean time series, is not it."""

SNAPPED = "snapped_fibers.npz"
SPHERES = ("lh_sphere_reg_lps.vtk", "rh_sphere_reg_lps.vtk")
"""The subject's registered spheres in the grid's frame, as ``from_snapped`` takes them."""

TIME_SERIES = "fc_ts.npz"
RUN_FOLDER = re.compile(r"^RUN\d+$")
"""The pipeline's per-run folders, each with its own ``fc_ts.npz``."""

NUISANCE_SCRIPT = "preprocessing/preproc_step5_fmri.sh"
"""Where the pipeline applies the nuisance model its ``FMRI_*`` settings describe."""


class ConversionError(SbciError, ValueError):
    """An input the conversion cannot use: the reason is the message."""


class ProfileError(SbciError, ValueError):
    """A pipeline profile that cannot be read, or that sets a value in a form not read here."""


# --- the profile ------------------------------------------------------------------------


_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


@dataclass(frozen=True)
class Profile:
    r"""A pipeline profile's settings, read as text and never run.

    The pipeline sources its profile in bash. Here only plain ``KEY=VALUE``
    lines are taken: a bare word, a quoted string or a one-line array
    ``KEY=(a b c)``, with ``export`` and a trailing comment allowed. A value
    that bash would expand (``$`` or a backquote), or any other form, is kept
    in :attr:`unread` and refused by :meth:`get`, since the stage scripts would
    still act on it.

    Examples
    --------
    >>> profile = Profile.parse("BANDWIDTH=0.005  # SC smoothing\nROIS=(4 8)\nX=$HOME\n")
    >>> profile.get("BANDWIDTH"), profile.get("ROIS")
    ('0.005', ('4', '8'))
    >>> sorted(profile.unread)
    ['X']
    """

    values: dict[str, Any] = field(default_factory=dict)
    unread: dict[str, str] = field(default_factory=dict)
    other: tuple[str, ...] = ()
    path: str = ""

    @classmethod
    def read(cls, path: str | Path) -> Profile:
        """Read a profile file."""
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise ProfileError(f"the profile {path} cannot be read: {exc}") from exc
        return cls.parse(text, path=str(path))

    @classmethod
    def parse(cls, text: str, path: str = "") -> Profile:
        """Read a profile from its text."""
        values: dict[str, Any] = {}
        unread: dict[str, str] = {}
        other: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = _ASSIGNMENT.match(line)
            if match is None:
                other.append(stripped)
                continue
            key, rest = match.groups()
            value = _plain_value(rest)
            if value is None:
                unread[key] = stripped
                values.pop(key, None)
            else:
                values[key] = value
                unread.pop(key, None)
        return cls(values, unread, tuple(other), path)

    def get(self, key: str) -> Any:
        """The value of ``key``, ``None`` when it is not set; refused when set in another form."""
        if key in self.unread:
            raise ProfileError(
                f"{self.path or 'the profile'} sets {key} as {self.unread[key]!r}, which is not "
                f"a plain value; write it as {key}=value"
            )
        return self.values.get(key)

    def settings(self, prefix: str) -> dict[str, Any]:
        """Every setting whose name starts with ``prefix``, in name order; refused as :meth:`get`.

        One set in a form not read here is refused rather than left out, which
        would describe the run without it.
        """
        keys = sorted(key for key in {*self.values, *self.unread} if key.startswith(prefix))
        return {key: self.get(key) for key in keys}


def _plain_value(rest: str) -> str | tuple[str, ...] | None:
    """The value of an assignment's right-hand side, or ``None`` when it is not a plain one."""
    if "$" in rest or "`" in rest:
        return None
    text = rest.strip()
    try:
        if text.startswith("("):
            inner, closing, after = text[1:].partition(")")
            if not closing or shlex.split(after, comments=True):
                return None
            return tuple(shlex.split(inner, comments=False))
        words = shlex.split(rest, comments=True)
    except ValueError:  # an unbalanced quote
        return None
    if len(words) > 1:  # KEY=a b runs b with KEY set; it assigns nothing
        return None
    return words[0] if words else ""


# --- names ------------------------------------------------------------------------------


def parse_sc_name(name: str) -> tuple[float, str] | None:
    """``(bandwidth, grid)`` from ``smoothed_sc_avg_<bandwidth>_<grid>.mat``, or ``None``.

    Examples
    --------
    >>> parse_sc_name("smoothed_sc_avg_0.01_ico4.mat")
    (0.01, 'ico4')
    >>> parse_sc_name("smoothed_sc_avg_1e-05_ico4.mat")
    (1e-05, 'ico4')
    >>> parse_sc_name("sub_sc_avg_0.005_ico4.mat") is None
    True
    """
    match = SC_NAME.match(name)
    if match is None:
        return None
    try:
        bandwidth = float(match["bandwidth"])
    except ValueError:
        return None
    if not (math.isfinite(bandwidth) and bandwidth > 0):
        return None
    return bandwidth, match["grid"]


def output_name(subject: str, modality: str, session: str | None = None) -> str:
    """A computational file's name, ``sub-<id>[_ses-<label>]_<modality>.h5``.

    The id goes through :func:`sbci.cohort.subject_id`, so ``01`` and
    ``sub-01`` name the same file. An id or session that would not read
    back from the name -- one holding a path separator, a ``ses-`` part, an
    empty label -- is refused.

    Examples
    --------
    >>> output_name("sub-01", "sc"), output_name("01", "fc", session="ses-2")
    ('sub-01_sc.h5', 'sub-01_ses-2_fc.h5')
    """
    from .cohort import parse_name, subject_id

    identifier = subject_id(subject)
    label = None
    if session is not None:
        label = str(session).strip()
        if label[:4].lower() == "ses-":
            label = label[4:]
        if not label:
            raise ValueError(f"session {session!r} has no label")
    if any(sep in identifier + (label or "") for sep in ("/", "\\", os.sep)):
        raise ValueError(f"{subject!r} cannot name a file: it holds a path separator")
    if identifier[4:].lower().startswith("ses-"):
        raise ValueError(
            f"{subject!r} is a session label, not a subject id; in a sub-<label>/ses-<label> "
            "layout the pipeline names its work folder after the session: pass the subject "
            "with --subject and the session with --session"
        )
    name = f"{identifier}_ses-{label}_{modality}.h5" if label else f"{identifier}_{modality}.h5"
    if parse_name(name) != (identifier, label, modality):
        given = f"subject {subject!r}" + (f" and session {session!r}" if label else "")
        raise ValueError(f"{given} would be read back from {name!r} as something else")
    return name


# --- reading ----------------------------------------------------------------------------


def read_matrix(path: str | Path, variable: str, n: int = spec.N_VERTICES) -> np.ndarray:
    """A pipeline ``.mat`` matrix as float64, refused unless it is ``n`` by ``n``."""
    import scipy.io
    from scipy import sparse

    path = Path(path)
    try:
        contents = scipy.io.loadmat(path, variable_names=[variable])
    except (OSError, ValueError, TypeError, NotImplementedError) as exc:
        raise ConversionError(f"{path.name} cannot be read as a MATLAB file: {exc}") from exc
    if variable not in contents:
        available = [k for k in scipy.io.whosmat(path) if not k[0].startswith("__")]
        raise ConversionError(
            f"{path.name} has no variable {variable!r}; it has {[k[0] for k in available]}"
        )
    value = contents.pop(variable)
    if sparse.issparse(value):
        value = value.toarray()
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
        raise ConversionError(f"{path.name}: {variable!r} is not a matrix of numbers")
    if value.shape != (n, n):
        raise ConversionError(
            f"{path.name}: {variable!r} is {value.shape}, expected ({n}, {n}), the ico4 grid"
        )
    return np.asarray(value, dtype=np.float64)


def mapping_areas(path: str | Path, n: int = spec.N_VERTICES) -> np.ndarray:
    """Vertex areas from a pipeline ``mapping_avg_ico4.npz``: high-resolution vertices per vertex.

    The mapping is an object array, so it is read with ``allow_pickle``: read
    only a mapping your own run wrote.
    """
    with np.load(path, allow_pickle=True) as contents:
        if "mapping" not in contents.files:
            raise ConversionError(f"{Path(path).name} has no 'mapping'")
        mapping = contents["mapping"]
    areas = np.array([np.asarray(m).size for m in mapping], dtype=np.float64)
    if areas.size != n:
        raise ConversionError(f"{Path(path).name}: {areas.size} vertices, expected {n}")
    if (areas == 0).any():
        raise ConversionError(f"{Path(path).name}: {(areas == 0).sum()} vertices have zero area")
    return areas


def bundled_areas() -> np.ndarray:
    """The package's vertex areas: fsaverage vertices per ico4 vertex, 327,684 in all."""
    from .io.cifti import vertex_areas

    return np.asarray(vertex_areas()[1], dtype=np.float64)


_VTK_TYPES = {
    "float": ">f4",
    "double": ">f8",
    "int": ">i4",
    "long": ">i8",
    "vtktypeint64": ">i8",
    "vtktypeint32": ">i4",
}


def read_vtk_points(path: str | Path) -> np.ndarray:
    """The ``POINTS`` of a legacy VTK file, ASCII or binary, as ``(n, 3)`` float64.

    What :meth:`sbci.smoothing.Endpoints.from_snapped` needs of the pipeline's
    registered spheres. Binary legacy files store big-endian values, as VTK
    writes them.
    """
    path = Path(path)
    with path.open("rb") as handle:
        first = handle.readline()
        if not first.lower().startswith(b"# vtk datafile"):
            raise ConversionError(f"{path.name} is not a legacy VTK file")
        handle.readline()  # the title
        form = handle.readline().strip().upper()
        if form not in (b"ASCII", b"BINARY"):
            raise ConversionError(
                f"{path.name}: {form.decode(errors='replace')!r} is not ASCII or BINARY"
            )
        while True:
            line = handle.readline()
            if not line:
                raise ConversionError(f"{path.name} has no POINTS")
            words = line.split()
            if words and words[0].upper() == b"POINTS":
                break
        try:
            count = int(words[1])
            kind = _VTK_TYPES[words[2].decode().lower()]
        except (IndexError, ValueError, KeyError) as exc:
            raise ConversionError(f"{path.name}: unreadable POINTS line {line!r}") from exc
        if form == b"BINARY":
            size = np.dtype(kind).itemsize * 3 * count
            raw = handle.read(size)
            if len(raw) != size:
                raise ConversionError(f"{path.name}: the POINTS end before {count} points")
            points = np.frombuffer(raw, dtype=kind).astype(np.float64)
        else:
            tokens = handle.read().split()[: 3 * count]
            if len(tokens) != 3 * count:
                raise ConversionError(f"{path.name}: the POINTS end before {count} points")
            try:
                points = np.array(tokens, dtype=np.float64)
            except ValueError as exc:
                raise ConversionError(f"{path.name}: the POINTS hold text: {exc}") from exc
    points = points.reshape(count, 3)
    if not np.isfinite(points).all():
        raise ConversionError(f"{path.name}: the POINTS hold values that are not finite")
    return points


def frames(path: str | Path) -> int:
    """The frames in a pipeline ``fc_ts.npz``: its time series' length, read from the headers.

    Only the two arrays' headers are read, not the series themselves, which
    run to gigabytes; the hemispheres have to agree.
    """
    import zipfile

    from numpy.lib import format as npy

    path = Path(path)
    found = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for key in ("lh_time_series", "rh_time_series"):
                with archive.open(f"{key}.npy") as member:
                    version = npy.read_magic(member)
                    if version == (1, 0):
                        shape = npy.read_array_header_1_0(member)[0]
                    elif version == (2, 0):
                        shape = npy.read_array_header_2_0(member)[0]
                    else:
                        raise ConversionError(f"{path.name}: {key} is in .npy format {version}")
                found[key] = shape
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ConversionError(
            f"{path.name}: the time series' headers cannot be read: {exc}"
        ) from exc
    lengths = {key: shape[1] if len(shape) == 2 else None for key, shape in found.items()}
    if None in lengths.values() or len(set(lengths.values())) != 1:
        raise ConversionError(f"{path.name}: the hemispheres' series disagree: {found}")
    return int(lengths["lh_time_series"])


def sha256(path: str | Path) -> str:
    """The SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# --- the conversion ---------------------------------------------------------------------


@dataclass
class Cleaned:
    """A matrix brought into the package's form, and what was taken out of it.

    ``diagonal`` and ``medial_wall`` are shares of the input's area-weighted
    mass, of the absolute values for FC; ``scale`` is what SC was divided by to
    reach unit mass (1 for FC).
    """

    data: np.ndarray
    diagonal: float
    medial_wall: float
    scale: float
    total: float

    def report(self) -> dict[str, float]:
        """The shares, as ``sbci convert`` prints them and the file records them."""
        return {
            "diagonal_share": self.diagonal,
            "medial_wall_share": self.medial_wall,
            "input_mass": self.total,
            "scale": self.scale,
        }


def clean(matrix: np.ndarray, modality: str, area: np.ndarray, mask: np.ndarray) -> Cleaned:
    """A pipeline matrix in the package's form: strict upper triangle, float32.

    ``matrix`` is the pipeline's ``n x n`` array: the upper triangle with the
    diagonal, the lower triangle empty. It is changed in place. A non-empty
    lower triangle, a value that is not finite, a matrix with no mass, and an
    SC with more than :data:`DIAGONAL_LIMIT` of its mass on the diagonal are
    refused with :class:`ConversionError`. The diagonal is dropped and the
    medial wall (``mask`` false) zeroed, each share of the area-weighted mass
    reported; SC is then divided by its area-weighted mass, so it integrates to
    one, and FC is left as correlations.

    Examples
    --------
    Four vertices, the last one medial wall, every connection 2:

    >>> sc = np.triu(np.full((4, 4), 2.0), 1)
    >>> out = clean(sc, "sc", np.ones(4), np.array([True, True, True, False]))
    >>> out.medial_wall, out.diagonal, out.scale
    (0.5, 0.0, 12.0)
    >>> out.data.tolist()
    [0.1666666716337204, 0.1666666716337204, 0.0, 0.1666666716337204, 0.0, 0.0]

    An FC keeps its values; its diagonal of ones is dropped and reported:

    >>> fc = np.triu(np.full((4, 4), 0.5), 1) + np.eye(4)
    >>> out = clean(fc, "fc", np.ones(4), np.ones(4, bool))
    >>> out.diagonal, out.medial_wall, out.data.tolist()
    (0.4, 0.0, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    Mass on an SC's diagonal is refused:

    >>> clean(np.triu(np.full((4, 4), 2.0)), "sc", np.ones(4), np.ones(4, bool))
    Traceback (most recent call last):
    ...
    sbci.convert.ConversionError: the diagonal holds 0.25 of the area-weighted mass, ...
    """
    if modality not in spec.MODALITIES:
        raise ValueError(f"modality must be one of {spec.MODALITIES}, got {modality!r}")
    n = matrix.shape[0]
    area = np.asarray(area, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if matrix.shape != (n, n) or area.shape != (n,) or mask.shape != (n,):
        raise ValueError(f"a {matrix.shape} matrix with {area.shape} areas and a {mask.shape} mask")

    bad = int(np.count_nonzero(~np.isfinite(matrix)))
    if bad:
        raise ConversionError(f"{bad:,} values are not finite (NaN or infinite)")
    # Row by row, so no second n x n array is made for the check.
    lower = sum(int(np.count_nonzero(matrix[i, :i])) for i in range(1, n))
    if lower:
        raise ConversionError(
            f"the lower triangle holds {lower:,} values; the pipeline writes the upper triangle "
            "with an empty lower one, so this is not its layout (a full symmetric matrix would "
            "count every connection twice)"
        )

    # Area-weighted shares of the symmetric matrix the triangle stands for, of the absolute
    # values: an SC is nonnegative, and an FC's signs would cancel in a plain sum.
    weights = np.abs(matrix) if modality == "fc" else matrix
    diagonal = np.diagonal(weights).copy()
    on_diagonal = float(area**2 @ diagonal)
    upper = float(area @ weights @ area) - on_diagonal
    cortex = np.where(mask, area, 0.0)
    kept = float(cortex @ weights @ cortex) - float(cortex**2 @ diagonal)
    del weights
    total = 2.0 * upper + on_diagonal
    if not total > 0:
        raise ConversionError("the matrix holds no connectivity: every value is zero")
    diagonal_share = on_diagonal / total
    wall_share = max(2.0 * (upper - kept) / total, 0.0)
    if modality == "sc" and diagonal_share > DIAGONAL_LIMIT:
        raise ConversionError(
            f"the diagonal holds {diagonal_share:.3g} of the area-weighted mass, more than "
            f"{DIAGONAL_LIMIT:g}; c3_main writes no self-connection, so this SC did not come "
            "from it, and dropping that mass would change it without a word"
        )

    np.fill_diagonal(matrix, 0.0)
    matrix[~mask, :] = 0.0
    matrix[:, ~mask] = 0.0
    scale = 1.0
    if modality == "sc":
        mass = 2.0 * float(area @ matrix @ area)
        if not mass > 0:
            raise ConversionError(
                "no structural connectivity is left once the medial wall is zeroed"
            )
        matrix /= mass
        scale = mass
    return Cleaned(
        data=_condensed(matrix),
        diagonal=diagonal_share,
        medial_wall=wall_share,
        scale=scale,
        total=total,
    )


def _condensed(matrix: np.ndarray) -> np.ndarray:
    """The strict upper triangle as float32, row by row: :func:`sbci.grid.to_condensed`'s order.

    Built a row at a time, so the index arrays of ``np.triu_indices``, 210 MB
    on the ico4 grid, are never made.
    """
    n = matrix.shape[0]
    out = np.empty(n * (n - 1) // 2, dtype=np.float32)
    start = 0
    for i in range(n - 1):
        stop = start + n - 1 - i
        out[start:stop] = matrix[i, i + 1 :]
        start = stop
    return out


# --- the command ------------------------------------------------------------------------


@dataclass
class Report:
    """What ``sbci convert`` did: printed on stdout as one JSON object."""

    subject: str = ""
    folder: str = ""
    out: str = ""
    written: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    removed: dict[str, dict[str, float]] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        """Record one check; a failed one is an error."""
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            self.errors.append(f"{name}: {detail}" if detail else name)
        return bool(passed)

    def warn(self, message: str) -> None:
        """Record a warning, and show it on stderr, which the pipeline keeps."""
        self.warnings.append(message)
        print(f"warning: {message}", file=sys.stderr)

    def input(self, modality: str, role: str, path: Path) -> None:
        """Record an input file with its size and SHA-256."""
        self.inputs.append(
            {
                "modality": modality,
                "role": role,
                "file": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    @property
    def status(self) -> str:
        """``written`` when everything available was written and passed, else ``failed``."""
        return "written" if self.written and not self.errors else "failed"

    def as_dict(self) -> dict[str, Any]:
        """The JSON object, in a fixed order."""
        return {
            "command": "sbci convert",
            "status": self.status,
            "subject": self.subject,
            "folder": self.folder,
            "out": self.out,
            "written": self.written,
            "skipped": self.skipped,
            "removed": self.removed,
            "checks": self.checks,
            "inputs": self.inputs,
            "versions": self.versions,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class UsageError(SbciError, ValueError):
    """The command line names something that is not there: exit status 2."""


def convert(
    folder: str | Path,
    subject: str,
    out: str | Path,
    pipeline_version: str | None = None,
    container_version: str | None = None,
    profile: str | Path | None = None,
    session: str | None = None,
    mapping: str | Path | None = None,
) -> Report:
    """Convert one subject's ``sbci_connectome`` folder; see the module notes.

    Returns the :class:`Report`; nothing is raised for a problem with the
    inputs, which the report's checks and errors carry. :class:`UsageError` is
    raised before anything is read when an argument names nothing.
    """
    folder = Path(folder)
    out = Path(out)
    report = Report(folder=str(folder.resolve()) if folder.exists() else str(folder))
    report.versions = {
        "sbci": __version__,
        "spec_version": spec.SPEC_VERSION,
        "pipeline_version": pipeline_version,
        "container_version": container_version,
    }
    if not folder.is_dir():
        raise UsageError(f"{folder} is not a folder")
    for name, given in (("--config", profile), ("--mapping", mapping)):
        if given is not None and not Path(given).is_file():
            raise UsageError(f"{name} {given} is not a file")
    try:
        names = {modality: output_name(subject, modality, session) for modality in spec.MODALITIES}
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    report.subject = names["sc"][: -len("_sc.h5")]
    if out.exists() and not out.is_dir():
        raise UsageError(f"--out {out} is a file, not a folder")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UsageError(f"--out {out} cannot be made: {exc}") from None
    report.out = str(out.resolve())

    if not pipeline_version:
        report.warn("no --pipeline-version: the files record the pipeline version as unrecorded")
    if not container_version:
        report.warn("no --container-version: the files record the container as unrecorded")
    elif "@sha256:" not in container_version:
        report.warn(
            f"the container {container_version!r} names no @sha256: digest, so the files "
            "cannot say which build of it ran"
        )
    versions = {
        "pipeline_version": pipeline_version or "unrecorded (not passed to sbci convert)",
        "container_version": container_version or "unrecorded (not passed to sbci convert)",
    }

    settings = Profile()
    if profile is not None:
        try:
            settings = Profile.read(profile)
            report.input("config", "profile", Path(profile))
        except ProfileError as exc:
            report.check("profile", False, str(exc))
            return report
        if settings.other:
            report.warn(
                f"the profile has {len(settings.other)} line(s) that are not plain assignments, "
                f"which are not followed, the first {settings.other[0]!r}"
            )

    from .atlas import cortex_mask

    mask = cortex_mask()
    try:
        area = bundled_areas()
    except (NotImplementedError, OSError) as exc:
        report.check("areas", False, f"the bundled vertex areas cannot be read: {exc}")
        return report
    if not _check_mapping(report, folder, mapping, area):
        return report  # on another grid, every matrix would be read into the wrong vertices

    _structural(report, folder, names["sc"], out, settings, versions, area, mask)
    _functional(report, folder, names["fc"], out, settings, versions, area, mask)
    if not report.written and not report.errors:
        report.check(
            "inputs",
            False,
            f"{folder} holds neither smoothed_sc_avg_<bandwidth>_{GRID}.mat nor "
            f"fc_avg_{GRID}.mat: nothing to convert",
        )
    return report


def _check_mapping(report: Report, folder: Path, mapping, area: np.ndarray) -> bool:
    """Compare the bundled areas with the run's mapping, where it can be found.

    Returns whether the conversion can go on: false when the mapping is there
    and cannot be read, or disagrees.
    """
    # Beside the connectivity, or where the pipeline's workflow keeps its grid:
    # <output>/files/work/<subject>/dwi_pipeline/sbci_connectome and <output>/files/grid.
    candidates = [Path(mapping)] if mapping is not None else [folder / f"mapping_avg_{GRID}.npz"]
    where = Path(os.path.abspath(folder))  # as named, not through links: the run's own layout
    if mapping is None and len(where.parents) > 3:
        candidates.append(where.parents[3] / "grid" / f"mapping_avg_{GRID}.npz")
    found = next((c for c in candidates if c.is_file()), None)
    if found is None:
        report.warn(
            f"no mapping_avg_{GRID}.npz beside the connectivity or in the run's grid folder, so "
            "the bundled vertex areas were not compared with the run's; pass --mapping to compare"
        )
        return True
    try:
        run_areas = mapping_areas(found)
    except (OSError, ValueError, SbciError) as exc:
        report.check("areas", False, f"{found.name} cannot be read: {exc}")
        return False
    report.input("both", "mapping", found)
    differ = int(np.count_nonzero(run_areas != area))
    return report.check(
        "areas",
        differ == 0,
        f"the run's {found.name} gives the bundled vertex areas at every vertex"
        if differ == 0
        else f"the run's {found.name} disagrees with the bundled vertex areas at {differ} "
        "vertices: the run is not on the package's grid",
    )


def _structural(report, folder, name, out, settings, versions, area, mask) -> None:
    """The SC file, if the folder holds the structural connectome."""
    candidates = sorted(p for p in folder.iterdir() if parse_sc_name(p.name) and p.is_file())
    if not candidates:
        report.skipped["sc"] = f"no smoothed_sc_avg_<bandwidth>_<grid>.mat in {folder}"
        return
    try:
        bandwidth = settings.get("BANDWIDTH")
        resolution = settings.get("RESOLUTION")
    except ProfileError as exc:
        report.check("sc: profile", False, str(exc))
        return
    if bandwidth is not None:
        try:
            wanted = float(bandwidth)
        except (TypeError, ValueError):
            report.check("sc: profile", False, f"BANDWIDTH={bandwidth!r} is not a number")
            return
        chosen = [p for p in candidates if math.isclose(parse_sc_name(p.name)[0], wanted)]
        if not chosen:
            found = ", ".join(p.name for p in candidates)
            report.check(
                "sc: bandwidth",
                False,
                f"the profile says BANDWIDTH={bandwidth}, but the structural connectome here is "
                f"{found}: the profile does not describe this run",
            )
            return
        candidates = chosen
    if len(candidates) > 1:
        report.check(
            "sc: inputs",
            False,
            f"{len(candidates)} structural connectomes here "
            f"({', '.join(p.name for p in candidates)}); pass the profile that names the "
            "bandwidth with --config",
        )
        return
    path = candidates[0]
    bandwidth_used, grid = parse_sc_name(path.name)
    if not report.check("sc: grid", *_grid(path.name, grid, resolution)):
        return
    if bandwidth is not None:
        report.check("sc: bandwidth", True, f"{bandwidth_used:g}, as the profile and the name say")

    endpoints, source = _endpoints(report, folder)
    report.input("sc", "connectivity", path)
    try:
        cleaned = clean(read_matrix(path, "sc"), "sc", area, mask)
    except ConversionError as exc:
        report.check("sc: connectivity", False, f"{path.name}: {exc}")
        return
    report.check(
        "sc: connectivity",
        True,
        f"{path.name}: diagonal {cleaned.diagonal:.3g} and medial wall "
        f"{cleaned.medial_wall:.3g} of the area-weighted mass removed; scaled to unit mass",
    )
    report.removed["sc"] = cleaned.report()
    from .metadata import template

    metadata = template(
        "sc",
        registration_reference="fsaverage",
        entry_point="precomputed",
        normalization="unit-mass",
        kernel="shk",
        bandwidth=bandwidth_used,
        streamline_count=int(endpoints.n_streamlines) if endpoints is not None else 0,
        streamline_weighting="none",
        endpoint_source=source,
        provenance=f"converted by sbci convert (sbci {__version__}) from the SBCI pipeline's "
        f"{path.name}; c3_main's spherical heat kernel at bandwidth {bandwidth_used:g}",
        inputs={item["file"]: item["sha256"] for item in report.inputs if item["modality"] != "fc"},
        conversion=cleaned.report(),
        **versions,
    )
    _write(report, "sc", out / name, cleaned.data, area, mask, metadata, endpoints)


def _grid(name: str, grid: str, resolution) -> tuple[bool, str]:
    """Whether a file's grid is the package's and the profile's, and what to say of it."""
    if grid != GRID:
        return False, f"{name} is on {grid}; the package reads {GRID} alone"
    if resolution not in (None, grid):
        return False, f"{name} is on {grid}, but the profile says RESOLUTION={resolution}"
    return True, f"{name} is on {GRID}"


def _endpoints(report: Report, folder: Path):
    """The streamline endpoints to store with the SC, and which branch they come from."""
    from .smoothing import Endpoints

    snapped = folder / SNAPPED
    spheres = [folder / name for name in SPHERES]
    if snapped.is_file() and all(s.is_file() for s in spheres):
        try:
            endpoints = Endpoints.from_snapped(snapped, *(read_vtk_points(s) for s in spheres))
            if endpoints.n_streamlines == 0:
                raise ConversionError("no streamline has both ends on the white surfaces")
        except (OSError, ValueError, KeyError, SbciError) as exc:
            report.check("sc: endpoints", False, f"{SNAPPED} on the registered spheres: {exc}")
        else:
            for path in (snapped, *spheres):
                report.input("sc", "endpoints", path)
            report.check(
                "sc: endpoints",
                True,
                f"{endpoints.n_streamlines:,} streamlines from {SNAPPED} on the registered "
                "spheres, the branch the SC was smoothed from",
            )
            return endpoints, f"{SNAPPED} on {' and '.join(SPHERES)}"
    elif snapped.is_file():
        missing = [s.name for s in spheres if not s.is_file()]
        report.warn(f"{SNAPPED} is here without {' and '.join(missing)}, so it cannot be placed")
    mesh = folder / f"mesh_intersections_{GRID}.mat"
    if mesh.is_file():
        try:
            endpoints = Endpoints.from_matlab(mesh)
        except (OSError, ValueError, KeyError, SbciError) as exc:
            report.check("sc: endpoints", False, f"{mesh.name}: {exc}")
            return None, "none"
        report.input("sc", "endpoints", mesh)
        report.warn(
            f"the endpoints stored are {mesh.name}'s, the pipeline's unsnapped branch: "
            "re-smoothing them does not reproduce this SC"
        )
        return endpoints, f"{mesh.name} (unsnapped: re-smoothing does not reproduce this SC)"
    report.warn("no streamline endpoints here: the SC file is written without them")
    return None, "none"


def _functional(report, folder, name, out, settings, versions, area, mask) -> None:
    """The FC file, if the folder holds the functional connectome."""
    candidates = sorted(p for p in folder.iterdir() if FC_NAME.match(p.name) and p.is_file())
    if not candidates:
        report.skipped["fc"] = f"no fc_avg_<grid>.mat in {folder}"
        return
    try:
        resolution = settings.get("RESOLUTION")
        fmri = settings.settings("FMRI_")
    except ProfileError as exc:
        report.check("fc: profile", False, str(exc))
        return
    on_grid = [p for p in candidates if FC_NAME.match(p.name)["grid"] == GRID]
    path = on_grid[0] if on_grid else candidates[0]
    if not report.check(
        "fc: grid", *_grid(path.name, FC_NAME.match(path.name)["grid"], resolution)
    ):
        return

    extra: dict[str, Any] = {}
    series = folder / TIME_SERIES
    if series.is_file():
        try:
            extra["fc_frames"] = frames(series)
        except ConversionError as exc:
            report.check("fc: frames", False, str(exc))
        else:
            report.input("fc", "time series", series)
            report.check("fc: frames", True, f"{extra['fc_frames']:,} frames in {TIME_SERIES}")
    else:
        report.warn(f"no {TIME_SERIES} here: the FC file does not record its frames")
    runs = sorted(
        p.name for p in folder.iterdir() if RUN_FOLDER.match(p.name) and (p / TIME_SERIES).is_file()
    )
    if runs:
        extra["fc_runs"] = runs

    report.input("fc", "connectivity", path)
    try:
        cleaned = clean(read_matrix(path, "fc"), "fc", area, mask)
    except ConversionError as exc:
        report.check("fc: connectivity", False, f"{path.name}: {exc}")
        return
    report.check(
        "fc: connectivity",
        True,
        f"{path.name}: diagonal {cleaned.diagonal:.3g} and medial wall {cleaned.medial_wall:.3g} "
        "of the area-weighted absolute values removed; correlations kept as they are",
    )
    report.removed["fc"] = cleaned.report()
    from .metadata import template

    metadata = template(
        "fc",
        registration_reference="fsaverage",
        entry_point="precomputed",
        normalization="none",
        fc_nuisance_model=nuisance_model(fmri, bool(settings.path)),
        provenance=f"converted by sbci convert (sbci {__version__}) from the SBCI pipeline's "
        f"{path.name}",
        inputs={item["file"]: item["sha256"] for item in report.inputs if item["modality"] != "sc"},
        conversion=cleaned.report(),
        **extra,
        **versions,
    )
    _write(report, "fc", out / name, cleaned.data, area, mask, metadata, None)


def nuisance_model(fmri: dict[str, Any], profile_given: bool = True) -> str:
    """The FC file's ``fc_nuisance_model``: the pipeline's step and the profile's ``FMRI_*`` values.

    Examples
    --------
    >>> nuisance_model({"FMRI_POLYFIT": "2", "FMRI_STC": "none"}).split(" with ")[1]
    'FMRI_POLYFIT=2, FMRI_STC=none; runs z-scored and concatenated'
    """
    where = f"SBCI pipeline FSFast nuisance regression ({NUISANCE_SCRIPT})"
    if fmri:
        values = ", ".join(
            f"{key}={' '.join(value) if isinstance(value, tuple) else value}"
            for key, value in fmri.items()
        )
        return f"{where} with {values}; runs z-scored and concatenated"
    if profile_given:
        return f"{where}; the profile sets no FMRI_ value; runs z-scored and concatenated"
    return f"{where}; settings unrecorded (no profile given); runs z-scored and concatenated"


def write_validated(path: str | Path, connectome) -> list:
    """Write ``connectome`` to ``path`` only if the written file passes every check.

    The file is saved under a temporary name beside ``path`` and checked by
    :func:`sbci.validate.validate_file`; it takes ``path``'s name only when
    every check passes, and is removed otherwise. Returns the checks. A write
    that fails raises, as :meth:`sbci.ContinuousConnectome.save` raises.
    """
    from .export import _target, _temporary
    from .validate import validate_file

    target = _target(Path(path))
    temporary = _temporary(target, f".{target.stem}.", ".h5")
    try:
        connectome.save(temporary)
        checks = validate_file(temporary)
        if all(check.passed for check in checks):
            os.replace(temporary, target)
        return checks
    finally:
        if temporary.exists():
            temporary.unlink()


def _write(report, modality, final: Path, data, area, mask, metadata, endpoints) -> None:
    """Write one file through :func:`write_validated`, recording its checks."""
    from .connectome import ContinuousConnectome

    try:
        connectome = ContinuousConnectome(data, area, mask, metadata, endpoints=endpoints)
        checks = write_validated(final, connectome)
    except (OSError, ValueError, SbciError) as exc:
        report.check(f"{modality}: write", False, f"{final}: {exc}")
        return
    for check in checks:
        report.checks.append(
            {
                "name": f"{modality}: validate {check.name}",
                "passed": bool(check.passed),
                "detail": str(check.detail),
            }
        )
    failed = [str(check) for check in checks if not check.passed]
    if failed:
        report.errors.append(
            f"{modality}: the file failed sbci validate and was not kept: {'; '.join(failed)}"
        )
        return
    report.written[modality] = str(final.resolve())


def _plain(value: Any) -> Any:
    """What the JSON encoder refuses, as plain values: NumPy scalars, and paths as text."""
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def command(args) -> int:
    """``sbci convert``: print the report as JSON; 0 when written and valid, 1, or 2 for usage."""
    try:
        report = convert(
            args.folder,
            args.subject,
            args.out,
            pipeline_version=args.pipeline_version,
            container_version=args.container_version,
            profile=args.config,
            session=args.session,
            mapping=args.mapping,
        )
    except UsageError as exc:
        print(f"sbci convert: error: {exc}", file=sys.stderr)
        print(json.dumps({"command": "sbci convert", "status": "usage", "errors": [str(exc)]}))
        return 2
    print(json.dumps(report.as_dict(), indent=2, default=_plain))
    for error in report.errors:
        print(f"error: {error}", file=sys.stderr)
    return 0 if report.status == "written" else 1

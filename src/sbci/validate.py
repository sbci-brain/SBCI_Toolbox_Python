"""The checks behind ``sbci validate <file>``.

Thirteen checks, each returning a human-readable line; the command exits
non-zero if any fails. CI runs them on synthetic files. PORTING.md records
which of them the released files have been run through: on the canonical
grid the mask and the areas have to be the grid's own exactly, and the stored
endpoints have to follow the corner rule of :data:`sbci.spec.ENDPOINTS_GROUP`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import grid, spec
from .connectome import COMPUTATIONAL_SUFFIXES, is_computational_name
from .errors import SbciError
from .io import read_hdf5
from .metadata import MetadataError
from .parcellation import CORRELATION_ROUNDING

CORRELATION_FLOOR = 1e-3
"""The least the largest magnitude of an FC file may be.

Correlations among the grid's cortical vertices reach it many times over:
the largest in :func:`sbci.example`'s FC is 0.96, and even pure noise of
19,200 frames, the HCP's four runs, puts the largest of the 11 million cortical
pairs near 0.04. A unit-mass SC density on this grid stays below 1e-6 (3e-9 in
the example's), so a density labelled FC fails, as does an FC file of zeros."""

CORNER_SHARE = 0.9
"""The share of endpoints whose stored vertex has to be the heaviest or the
nearest corner of its triangle (the corner rule, :data:`sbci.spec.ENDPOINTS_GROUP`).

The package's writers store the heaviest corner, and their files score 100%:
:func:`sbci.example`'s, and endpoints placed as
:meth:`sbci.smoothing.Endpoints.from_snapped` places the pipeline's. A writer
that stores the nearest vertex scores 100% as well, and the MATLAB pipeline's
``mesh_intersections_ico4.mat`` has its heaviest corner stored for 97.16%
(PORTING.md item 6). Weights stored in another corner order than the face
list's score 1% to 3%, though every such endpoint still rebuilds within one
edge of its vertex."""

_WEIGHT_TIE = 1e-6
"""Weights within this of the largest count as the largest: they are stored as
float32, which rounds weights near one half by 3e-8."""

_COSINE_TIE = 1e-9
"""Corners whose cosines with the position lie within this of the nearest's
count as the nearest: less than a thousandth of a degree."""

_BLOCK = 1 << 18
"""Endpoints judged at a time, so a file of 1.6 million keeps its temporaries small."""


@dataclass
class Check:
    """One validation result."""

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}" + (f": {self.detail}" if self.detail else "")


def _named_modality(path: Path) -> str | None:
    """The modality a computational file's name says before its extension, ``sc`` or ``fc``.

    ``sub-01_ses-1_fc.h5`` says ``fc``; ``example.h5`` says none.
    """
    name = path.name
    for suffix in COMPUTATIONAL_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].lower() in spec.MODALITIES:
        return parts[1].lower()
    return None


def validate_file(path: str | Path, tolerance: float = 1e-5) -> list[Check]:
    """Check one computational file against the specification.

    The checks, in order:

    ``readable``
        The file opens and its parts read.
    ``name``
        It is named as :meth:`sbci.ContinuousConnectome.load` reads it, and a
        name ending in ``_sc`` or ``_fc`` says the modality its metadata says.
    ``metadata``
        Complete and canonical (:meth:`sbci.metadata.Metadata.validate`): the
        version a readable one, and each fixed key's value one the format has.
    ``grid``, ``shapes``
        The connectivity is the ico4 grid's triangle, and the area, mask and
        coordinates the size of that grid.
    ``storage``
        Stored as the strict upper triangle in float32.
    ``finite``, ``nonnegativity``, ``correlation``
        Every value finite; SC nonnegative; FC within ``[-1, 1]`` to rounding
        (:data:`sbci.parcellation.CORRELATION_ROUNDING`) and reaching
        :data:`CORRELATION_FLOOR` somewhere, as correlations do.
    ``area``
        Positive, finite, and on the ico4 grid exactly the grid's own vertex
        areas (:func:`sbci.io.cifti.vertex_areas`).
    ``unit mass``
        For SC, the area-weighted mass is one.
    ``mask``
        On the ico4 grid exactly the grid's medial wall
        (:func:`sbci.atlas.cortex_mask`), and the wall empty: the
        connectivity's finite entries there come to no more than
        ``tolerance`` of their total.
    ``endpoints``
        Each stored endpoint's position, rebuilt from its triangle, lies
        within one edge of its stored vertex, and that vertex is the heaviest
        or the nearest corner of the triangle for :data:`CORNER_SHARE` of them.

    Tolerances are relative to the file's own scale where a scale exists.
    Everything ``load`` refuses fails some check here; the converse does not
    hold, since ``load`` checks few of the values -- the mask's, the
    endpoints' -- and none of the connectivity's.
    """
    path = Path(path)
    checks: list[Check] = []
    if not Path(path).exists():
        return [Check("readable", False, f"no such file: {path}")]

    try:
        parts = read_hdf5(path)
        import h5py

        with h5py.File(path, "r") as handle:
            stored_dtype = handle["connectivity"].dtype
    except (OSError, KeyError, SbciError, ValueError, TypeError) as exc:
        return [Check("readable", False, str(exc))]
    checks.append(Check("readable", True))

    metadata = parts["metadata"]

    # load() decides what kind of file it has from the name, before reading; a name
    # that says the other modality is a contradiction, as load_cohort reads it.
    said, held = _named_modality(path), metadata.get("included_connections")
    if not is_computational_name(path):
        checks.append(
            Check("name", False, f"{path.name!r} would not load; expected {COMPUTATIONAL_SUFFIXES}")
        )
    elif said is not None and held in spec.MODALITIES and said != held:
        checks.append(
            Check("name", False, f"{path.name!r} says {said}, but its metadata says {held}")
        )
    else:
        checks.append(Check("name", True))

    try:
        metadata.validate()
        checks.append(Check("metadata", True))
    except MetadataError as exc:
        checks.append(Check("metadata", False, str(exc)))

    data = parts["data"]
    area = parts["area"]
    mask = parts["mask"]

    # Grid: the condensed length has to correspond to the canonical grid.
    try:
        n = grid.n_from_condensed(data.size)
        ok = n == spec.N_VERTICES
        checks.append(Check("grid", ok, "" if ok else f"{n} vertices, expected {spec.N_VERTICES}"))
    except ValueError as exc:
        n = None
        checks.append(Check("grid", False, str(exc)))

    # The arrays have to agree on the vertex count, as load() requires of them;
    # the coordinates are optional but, when present, one triple per vertex.
    coords = parts["coords"]
    shapes_agree = n is not None and data.ndim == 1 and area.shape == (n,) and mask.shape == (n,)
    if n is None:
        checks.append(Check("shapes", False, "no vertex count to compare the arrays against"))
    elif area.size != n or mask.size != n:
        checks.append(
            Check(
                "shapes",
                False,
                f"connectivity implies {n} vertices but area has {area.size} and mask has "
                f"{mask.size}",
            )
        )
    elif not shapes_agree:
        checks.append(
            Check(
                "shapes",
                False,
                f"connectivity is stored {data.shape}, area {area.shape} and mask "
                f"{mask.shape}, where each is flat",
            )
        )
    elif coords is not None and np.shape(coords) != (n, 3):
        checks.append(
            Check("shapes", False, f"coordinates are {np.shape(coords)}, expected {(n, 3)}")
        )
    else:
        checks.append(Check("shapes", True))

    # The stored form: storing only the upper triangle makes the matrix symmetric,
    # so the check is that the stored form is the one the spec names.
    convention = metadata.get("storage_convention", "")
    checks.append(
        Check(
            "storage",
            convention == spec.STORAGE_CONVENTION and stored_dtype == np.float32,
            f"storage_convention={convention!r}, stored dtype={stored_dtype}",
        )
    )

    # Every value has to be a number: NaN passes every inequality below.
    finite = np.isfinite(data)
    non_finite = int(np.count_nonzero(~finite))
    checks.append(Check("finite", non_finite == 0, f"{non_finite} non-finite entries"))

    # Nonnegativity applies to SC only; FC is signed by construction.
    if metadata.get("included_connections") == "sc":
        negative = int(np.count_nonzero(data < 0))
        checks.append(Check("nonnegativity", negative == 0, f"{negative} negative entries"))
    else:
        checks.append(Check("nonnegativity", True, "skipped: FC is signed"))

    # FC holds correlations, which rounding may carry a little past 1 and no further:
    # a file of Fisher z values, or of correlations rescaled, passed every check, and
    # to_atlas's Fisher z average clipped it without a word. Correlations also reach
    # well past zero somewhere, which an SC density labelled FC never does.
    if metadata.get("included_connections") == "fc":
        magnitude = np.abs(np.where(finite, data, 0.0))
        outside = int(np.count_nonzero(magnitude > 1.0 + CORRELATION_ROUNDING))
        largest = float(magnitude.max()) if magnitude.size else 0.0
        detail = f"{outside} entries outside [-1, 1]"
        if outside:
            detail += f", up to {largest:.7g} in magnitude"
        elif largest < CORRELATION_FLOOR:
            detail += (
                f", but none reaches {CORRELATION_FLOOR:g} in magnitude (the largest is "
                f"{largest:.3g}): correlations do, so these are not correlations"
            )
        checks.append(Check("correlation", outside == 0 and largest >= CORRELATION_FLOOR, detail))
    else:
        checks.append(Check("correlation", True, "skipped: only FC holds correlations"))

    # Area weights: positive, finite, and -- on the canonical grid -- the grid's own,
    # fsaverage vertices per ico4 vertex (SPEC_QUESTIONS item 3). to_atlas weighs by
    # the file's areas, save_map and region_means by the grid's: uniform areas of the
    # right total passed, and gave the two different answers.
    if area.size and np.isfinite(area).all() and (area > 0).all():
        if area.shape == (spec.N_VERTICES,) and n == spec.N_VERTICES:
            from .io.cifti import vertex_areas

            canonical = vertex_areas()[1]
            differ = area != canonical
            if differ.any():
                checks.append(
                    Check(
                        "area",
                        False,
                        f"{int(differ.sum())} of {area.size} vertex areas differ from the "
                        "grid's own (sbci.io.cifti.vertex_areas), by up to "
                        f"{float(np.abs(area - canonical).max()):.6g}; sum {area.sum():.6g}, "
                        f"the grid's {spec.AREA_TOTAL}",
                    )
                )
            else:
                checks.append(
                    Check("area", True, f"the grid's own vertex areas, sum {area.sum():.6g}")
                )
        else:
            checks.append(Check("area", True, "positive and finite"))
    else:
        checks.append(Check("area", False, "area weights must be positive and finite"))

    dense = None
    if shapes_agree:
        dense = grid.to_dense(data.astype(np.float64), n)

    # Unit mass: the area-weighted total density integrates to one. This is a
    # property of a density, so it applies to SC only -- FC is a field of
    # correlations whose area-weighted total is an arbitrary number near zero,
    # and rescaling it to one would destroy its units.
    if metadata.get("included_connections") != "sc":
        checks.append(Check("unit mass", True, "skipped: FC is not a density"))
    elif dense is not None:
        total = float(area @ dense @ area)
        ok = abs(total - 1.0) <= tolerance
        checks.append(Check("unit mass", ok, f"total mass {total:.6g}"))
    else:
        checks.append(Check("unit mass", False, "the arrays' shapes do not match the grid"))

    # Mask: on the canonical grid the grid's own medial wall, which to_atlas, the
    # exports and region_means all take a file's mask to be; and the wall must carry
    # no connectivity. The leak is judged against the file's own total, since a
    # unit-mass density on the ico4 grid has entries of order 1e-11 and an absolute
    # tolerance would pass anything, and over the finite entries, which the finite
    # check judges. A mask that keeps nothing leaves nothing to leak: an FC file of
    # zeros with no cortex passed every check.
    if dense is not None:
        if mask.all():
            checks.append(Check("mask", False, "no vertices are masked out"))
        elif not mask.any():
            checks.append(Check("mask", False, "no vertex is cortex: the mask keeps nothing"))
        elif n == spec.N_VERTICES and (mask != _canonical_mask()).any():
            canonical = _canonical_mask()
            called_cortex = int(np.count_nonzero(mask & ~canonical))
            called_wall = int(np.count_nonzero(~mask & canonical))
            checks.append(
                Check(
                    "mask",
                    False,
                    f"the mask is not the grid's medial wall (sbci.atlas.cortex_mask): "
                    f"{called_cortex} wall vertices called cortex and {called_wall} cortical "
                    "vertices called wall",
                )
            )
        else:
            wall_rows = dense[~mask]
            leaked = float(np.abs(wall_rows[np.isfinite(wall_rows)]).sum())
            # each pair appears twice in the dense matrix, once in the stored triangle
            total_abs = 2.0 * float(np.abs(data[finite]).sum(dtype=np.float64))
            ok = leaked <= tolerance * total_abs
            share = leaked / total_abs if total_abs > 0 else 0.0
            detail = f"medial wall carries {share:.3g} of the total"
            if non_finite:
                detail += f", the {non_finite} non-finite entries left out"
            checks.append(Check("mask", ok, detail))
    else:
        checks.append(Check("mask", False, "the arrays' shapes do not match the grid"))

    # Endpoints: the position rebuilt from an endpoint's triangle lies within one edge of
    # its stored vertex when the triangles index the grid's own face list; against a
    # face list in another order it lands about 85 degrees away (SPEC_QUESTIONS.md item 13).
    # Within one edge, weights stored in another corner order still pass: the corner
    # rule catches them, the stored vertex being their heaviest or nearest corner.
    endpoints = parts["endpoints"]
    if endpoints is None or not endpoints.has_positions:
        checks.append(Check("endpoints", True, "skipped: no endpoint positions"))
    elif n != spec.N_VERTICES:
        checks.append(
            Check("endpoints", True, "skipped: not on the ico4 grid, whose faces they index")
        )
    else:
        checks.append(_endpoints_check(endpoints))

    return checks


def _canonical_mask() -> np.ndarray:
    from .atlas import cortex_mask

    return cortex_mask()


def _endpoints_check(endpoints) -> Check:
    """The ``endpoints`` check: one edge from the stored vertex, and the corner rule."""
    from .smoothing import _vertex_offsets
    from .surface import load_surface

    surface = load_surface("sphere")
    try:
        points_in, points_out, offsets, edge = _vertex_offsets(endpoints, surface)
    except ValueError as exc:
        return Check("endpoints", False, str(exc))
    far = int(np.count_nonzero(offsets > edge))
    detail = (
        f"{far:,} of {offsets.size:,} endpoints rebuild farther than one edge "
        f"({edge:.2f} degrees) from their stored vertex, up to {offsets.max():.1f}"
        if far
        else f"every endpoint rebuilds within one edge ({edge:.2f} degrees) of its "
        f"stored vertex, at most {offsets.max():.2f}"
    )
    shares = _corner_shares(endpoints, points_in, points_out, surface)
    agree = shares["heaviest or nearest"]
    detail += (
        f"; the stored vertex is its triangle's heaviest or nearest corner for "
        f"{100 * agree:.1f}% (heaviest {100 * shares['heaviest']:.1f}%, nearest "
        f"{100 * shares['nearest']:.1f}%, a corner {100 * shares['corner']:.1f}%)"
    )
    if agree < CORNER_SHARE:
        detail += (
            f", under the {100 * CORNER_SHARE:.0f}% the corner rule needs: the weights are not "
            "in the corner order of the grid's face list (sbci.spec.ENDPOINTS_GROUP)"
        )
    return Check("endpoints", far == 0 and agree >= CORNER_SHARE, detail)


def _corner_shares(endpoints, points_in, points_out, surface) -> dict[str, float]:
    """The shares of endpoints whose stored vertex is a corner of its triangle, and which corner.

    ``"corner"``: a corner at all; ``"heaviest"``: the corner of the largest
    weight; ``"nearest"``: the corner nearest the rebuilt position;
    ``"heaviest or nearest"``: either, the share the corner rule judges.
    """
    from .smoothing import Endpoints

    faces = np.asarray(surface.faces, dtype=np.int64)
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    per_hemi = faces.shape[0] // 2
    counts = dict.fromkeys(("corner", "heaviest", "nearest", "heaviest or nearest"), 0)
    total = 0
    for triangle, bary, surf, vertex, points in (
        (
            endpoints.tri_in,
            endpoints.bary_in,
            endpoints.surf_in,
            endpoints.global_vertex_in,
            points_in,
        ),
        (
            endpoints.tri_out,
            endpoints.bary_out,
            endpoints.surf_out,
            endpoints.global_vertex_out,
            points_out,
        ),
    ):
        for start in range(0, len(vertex), _BLOCK):
            part = slice(start, start + _BLOCK)
            index = np.asarray(triangle[part], dtype=np.int64)
            index = index + Endpoints.global_hemisphere_offset(surf[part], per_hemi)
            corners = faces[index]
            stored = corners == np.asarray(vertex[part])[:, None]
            weights = np.asarray(bary[part], dtype=np.float64)
            heaviest = (
                np.where(stored, weights, -np.inf).max(axis=1) >= weights.max(axis=1) - _WEIGHT_TIE
            )
            cosine = np.einsum("sj,skj->sk", points[part], unit[corners])
            nearest = (
                np.where(stored, cosine, -np.inf).max(axis=1) >= cosine.max(axis=1) - _COSINE_TIE
            )
            counts["corner"] += int(stored.any(axis=1).sum())
            counts["heaviest"] += int(heaviest.sum())
            counts["nearest"] += int(nearest.sum())
            counts["heaviest or nearest"] += int((heaviest | nearest).sum())
            total += len(corners)
    return {key: value / total if total else 1.0 for key, value in counts.items()}

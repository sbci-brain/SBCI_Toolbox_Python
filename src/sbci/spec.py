"""The WP1 file specification, as constants.

Everything in the package reads its conventions from here so that a change to
the specification is a change to one file.

The version rule
----------------
A file's ``spec_version`` is ``MAJOR.MINOR.PATCH``: three whole numbers without
leading zeros, then optionally ``-draft``, the one tag there is
(:data:`VERSION_PATTERN`). ``MAJOR.MINOR`` names the format: two files of one
``MAJOR.MINOR`` hold the same datasets with the same meanings, and ``PATCH``
changes the specification's wording only. A build reads the formats in
:data:`READABLE_VERSIONS`, whatever a file's ``PATCH`` and tag, and writes
:data:`SPEC_VERSION`. A new format adds its ``MAJOR.MINOR`` there and keeps the
old ones it still reads.

.. warning::
   ``SPEC_VERSION`` is a *draft*. WP1 has not frozen the specification yet, and
   the open questions are listed in ``SPEC_QUESTIONS.md`` at the repository
   root. Nothing here should be treated as final until that document is empty
   and ``SPEC_VERSION`` loses its ``-draft`` suffix. The released young adults
   (doi:10.15139/S3/NDAJYX, version 1.0) are ``0.1.0-draft`` files, format
   ``0.1``: they load under any ``SPEC_VERSION`` while ``"0.1"`` is in
   :data:`READABLE_VERSIONS`.
"""

from __future__ import annotations

SPEC_VERSION = "0.1.0-draft"
"""Version of the on-disk format this build of the package writes."""

READABLE_VERSIONS = frozenset({"0.1"})
"""The formats this build reads, as ``MAJOR.MINOR``.

A file loads when its ``spec_version`` names one of them, whatever its
``PATCH`` and tag: ``0.1.0-draft``, ``0.1.0`` and ``0.1.7`` all read as format
``0.1``. Listed rather than taken from :data:`SPEC_VERSION`, so that a new
``SPEC_VERSION`` cannot lock out files already published; the format
``SPEC_VERSION`` writes is always among them."""

VERSION_PATTERN = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-draft)?"
"""What a ``spec_version`` matches, whole: ``0.1.0`` and ``0.1.0-draft`` do; the
number ``0.1``, the text ``"0.1"``, ``0.1.0-final`` and ``0.1.x.y`` do not."""

# --- Computational grid (WP1) ---------------------------------------------

GRID = "fsaverage-ico4"
"""Canonical computational grid. The ``_0.99`` / ``_0.94`` reduction-fraction
grids and the 4121-vertex alignment grid are retired and are not read."""

N_VERTICES_PER_HEMI = 2562
N_VERTICES = 5124
"""Total vertices, left hemisphere first, then right."""

N_FACES_PER_HEMI = 5120
N_FACES = 10240
"""Triangles of the grid mesh, left hemisphere first, then right.

The face list is the pipeline's ``?h_grid_avg_ico4.m``, in its order, which
every bundled surface carries (SPEC_QUESTIONS.md item 13): triangle indices in
a file count in it, and the corners of a face are in its order."""

AREA_TOTAL = 327_684
"""Sum of the per-vertex area weights on the canonical grid.

Areas are counted in fsaverage vertices (163,842 per hemisphere), so they are
integers per ico4 vertex and total the whole fsaverage sphere; the ico4-to-fsLR
overlap matrix has the same column sums (SPEC_QUESTIONS.md item 3).
"""

HEMISPHERE_ORDER = ("L", "R")

# --- Storage convention ----------------------------------------------------

TRIANGLE = "upper"
DIAGONAL_INCLUDED = False
"""Connectivity is stored as the strict upper triangle (k=1), float32.

The self-connectivity diagonal is excluded: the legacy ``parcellate_sc.m``
removes it before aggregating, so carrying it would change every downstream
number. See ``SPEC_QUESTIONS.md`` item 2 -- WP1 must confirm.
"""

DTYPE = "float32"

STORAGE_CONVENTION = f"{TRIANGLE}-triangular-{DTYPE}"
"""The ``storage_convention`` every file names: ``upper-triangular-float32``."""

N_UPPER = N_VERTICES * (N_VERTICES - 1) // 2
"""Length of the condensed connectivity vector: 13,125,126 float32 ~= 50 MB."""

# --- Streamline endpoints --------------------------------------------------

ENDPOINTS_GROUP = "endpoints"
"""Optional group in the computational file holding the streamline endpoints
that ``ContinuousConnectome.smooth()`` re-smooths from.

Endpoints live in the same file as the connectivity rather than in a sibling,
so that a connectome and the endpoints it was built from cannot be separated,
mismatched or versioned apart. The group is optional: a file without it is
valid and simply cannot be re-smoothed.

Vertex and triangle indices are **global and zero-based** -- 0 to 5123 over the
two hemispheres for vertices, 0 to 10239 for triangles -- so the hemisphere is
implied by the index and cannot disagree with a separate field.

The barycentric datasets are optional within the group. They give each
endpoint's continuous position inside its triangle, which a kernel evaluated
off-grid needs; without them an endpoint is only known to the nearest vertex.

**The corner rule.** ``barycentric_in[s, k]`` is the weight of corner ``k`` of
face ``triangle_in[s]``, its corners in the order the face list gives them
(:data:`N_FACES`), and the endpoint lies at ``sum_k w_k x_k`` brought back
onto the sphere. ``vertex_in[s]`` is a corner of that face: the corner of the
largest weight, as the package's writers store it, or the corner nearest the
position, as a writer that snaps each endpoint to its nearest vertex stores
it; the two differ for about 3% of endpoints. The same holds for ``_out``.
``sbci validate`` checks the rule.
"""

ENDPOINT_DATASETS = ("vertex_in", "vertex_out")
"""Required inside :data:`ENDPOINTS_GROUP`."""

ENDPOINT_OPTIONAL_DATASETS = (
    "triangle_in",
    "triangle_out",
    "barycentric_in",
    "barycentric_out",
)
"""Present together or not at all."""

# --- Exchange format (WP1) -------------------------------------------------

EXCHANGE_SPACE = "fsLR"
EXCHANGE_DENSITY = "32k"
EXCHANGE_SUFFIX = ".dconn.nii"
"""How an exchange file's name ends: write-only, so :meth:`sbci.ContinuousConnectome.load`
refuses a name ending in it (or in it and ``.gz``) as one."""

# --- Metadata contract -----------------------------------------------------

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "grid",
    "hemisphere_order",
    "mask",
    "area_weights",
    "storage_convention",
    "included_connections",
    "streamline_count",
    "streamline_weighting",
    "normalization",
    "kernel",
    "bandwidth",
    "fc_nuisance_model",
    "registration_reference",
    "pipeline_version",
    "container_version",
)
"""Keys every file must carry. ``ContinuousConnectome.load`` refuses a file
that is missing any of them -- WP2 requirement, not a warning."""

SC_ONLY_KEYS = ("streamline_count", "streamline_weighting", "kernel", "bandwidth")
FC_ONLY_KEYS = ("fc_nuisance_model",)
"""Keys of one modality alone: a file of the other modality carries none of
them, or carries them empty (``null``)."""

SC_OPTIONAL_KEYS = ("nu",)
"""Keys an SC file carries only for the kernel they belong to: ``nu``, the Matern
smoothness, written with ``kernel="matern"`` alone. A Matern file without it was
made before it was recorded, when 3 was the only value used."""

MODALITIES = ("sc", "fc")

MASK_CONVENTION = "medial-wall"
"""The ``mask`` every file names: ``/mask`` is true on cortex and false on the
medial wall, the wall :func:`sbci.atlas.cortex_mask` gives."""

AREA_CONVENTION = "vertex-area"
"""The ``area_weights`` every file names: ``/area`` holds each grid vertex's area
in fsaverage vertices, the counts summing to :data:`AREA_TOTAL`."""

NORMALIZATION = {"sc": "unit-mass", "fc": "none"}
"""The ``normalization`` of each modality: SC is a density whose area-weighted
mass is one, FC correlations stored as they are."""

KERNELS = ("shk", "rdk", "matern")
"""The ``kernel`` an SC file names, in lowercase: :data:`sbci.smoothing.KERNELS`."""

ENTRY_POINTS = ("raw", "preprocessed", "fastpath", "precomputed")
"""Values of the optional ``entry_point`` metadata key: where the data entered
the pipeline. The build tools write ``preprocessed`` (built from pipeline
output) and ``precomputed`` (a legacy file imported). Nothing reads the key,
and the reader does not check it."""

"""Kernel smoothing on the cortical surface: streamline endpoints to a continuous connectome.

Three kernels (:data:`KERNELS`). ``shk``, the default, puts a bump around each
endpoint's exact position on the registered sphere and reproduces the released
files: it is the spherical kernel ``concon`` applies. ``rdk`` and ``matern``
smooth the endpoints' vertex counts with the white surface's Laplace-Beltrami
basis, as ``K A K'``: the Riemannian diffusion and Matern kernels of
``SBCI_Toolkit/concon_estimate/compute_diffusion_kernel_matrix.m``,
``compute_matern_kernel_matrix.m`` and the driver
``rdk_smoothed_concon_compute.m``.

The rdk and matern method
-------------------------
Streamline endpoints give a count matrix ``A`` on the grid, one block per pair
of hemispheres. With a hemisphere's Laplace-Beltrami eigenpairs
``(Lambda, Phi)``, the kernel is::

    K = Phi diag(rho(Lambda)) Phi'

``rho`` is the only thing that differs between the two kernels: the diffusion
kernel uses ``exp(-kappa^2 Lambda / 2)`` and the Matern kernel
``(2 nu / kappa^2 + Lambda)^(-nu-1)``. Column ``j`` of ``K`` is the bump an
endpoint at vertex ``j`` spreads over the surface. Each column is scaled to
integrate to one under the file's vertex weights, ``Kt = K / (area @ K)``, and
the density is ``Kt A Kt'``: every streamline end carries the same mass
wherever it lands.

The basis is built here, on the grid itself (:func:`laplace_beltrami_basis`):
the cotangent Laplacian of the bundled ico4 white surface -- fsaverage's white
surface, averaged over each grid vertex -- on its intrinsic Delaunay
triangulation, with each vertex's lumped area, all 2,562 modes of a hemisphere,
orthonormal under those areas. ``K`` is then a proper smoothing kernel:
nonnegative, local, integrating to one over the surface, and shrinking to a
point mass at its vertex as the bandwidth does. Building it takes under half a
minute for both hemispheres on one core, once a session, and needs no download.

The reference's own basis is not that, and this module no longer uses it unless
asked to. ``EV_LBO_ds_ico4_{L,R}.mat`` hold the first 2,562 eigenvectors of
fsaverage's white surface at ico6, sampled at the ico4 vertices and given unit
Euclidean length. They are not orthonormal on ico4 (``U' U`` is off the
identity by up to 0.12), so ``K`` does not shrink to a point mass as the
bandwidth does, but to ``U U'``. At the reference's default bandwidth about a
third of the kernel's mass is negative and half lies beyond 20 degrees; the
reference clips the density at zero. :func:`find_basis` and ``eigenpairs=``
still read those files, for comparisons with the MATLAB, and :func:`smooth`
warns when a basis makes a signed kernel.
"""

from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache, partial

import numpy as np

from . import spec
from .errors import FormatError, MissingDataError

KERNELS = ("shk", "rdk", "matern")
"""The kernel vocabulary, following the decision recorded as SPEC_QUESTIONS.md
item 10.

``shk``
    The spherical kernel ``concon`` applies (Moyer et al., MICCAI 2016), the
    default, and the one that produced every released ``smoothed_sc_avg_*.mat``.
    Its bandwidth is ``sigma``, a diffusion time on the unit sphere. Reproduces
    ``c3_main`` at r = 1.000000 and scale 1.000000 at full scale on the
    tutorial subject, but for the grid's south pole, which the binary never
    writes and this does; the amplitude offset an earlier version carried was
    the binary's lookup-table quantization, which :class:`KernelTable`
    reproduces -- see PORTING.md item 6. Its name, and that of
    :func:`spherical_heat_kernel`, is historical: it is not a heat kernel, and
    that function's docstring explains the weight.
``rdk``
    Riemannian diffusion kernel (Wang, Li and Zhang, Imaging Neuroscience 3, 2025,
    doi:10.1162/imag.a.912): the heat kernel of the white surface, from its
    Laplace-Beltrami eigenpairs (:func:`laplace_beltrami_basis`). Bandwidth
    ``kappa``, in mm on the white surface. :func:`diffusion_kernel` is the
    reference's formula, checked against the MATLAB on the MATLAB's own basis.
``matern``
    Matern kernel on the same basis, from ``compute_matern_kernel_matrix.m``.
    Bandwidth ``kappa``, in mm, plus a smoothness ``nu``. :func:`matern_kernel`.

``bandwidth`` therefore means different things for different kernels, and has
to be read together with ``kernel`` in the metadata.
"""

TRUNCATION_FLOOR = 1e-30
"""Kernel weights below this fraction of the largest are set to zero.

The MATLAB's floor is absolute, on a diffusion kernel whose largest weight,
at the eigenvalue 0, is 1; for it the two are the same. Relative, the floor
also serves the Matern kernel, whose largest weight is
``(2 nu / kappa^2)^(-nu-1)``, 1.3e-174 at ``nu`` 100 and ``kappa`` 1.95:
the absolute floor zeroed every weight from ``nu`` 15 up at small ``kappa``."""

DEFAULT_NU = 3.0
"""The Matern smoothness :func:`smooth` uses when given none: the ``nu = 3`` of
``rdk_smoothed_concon_compute.m``, which names 1, 2 and 3 as the values to use."""

DEFAULT_KAPPA = {"rdk": 5.5, "matern": 5.2}
"""The bandwidth :func:`smooth` uses for ``rdk`` and ``matern`` when given none, in mm.

Each makes its kernel average over as many vertices as ``shk`` does at its
default sigma of 0.005: an effective number of independent vertices,
``K_eff``, of 302.7 for ``shk`` over the cortex, against 306 for ``rdk`` at
5.5 mm and 299 for ``matern`` at 5.2 mm. ``K_eff`` is the measure of the
paper's Section 6.3: the cortex's weight over the mean weight one smoothed
value averages over, ``(sum a w)^2 / sum a w^2``, with the file's vertex
weights ``a``. The three default kernels thus smooth at one scale.
``matern``'s matches at its default ``nu``, 3; another ``nu`` keeps the
bandwidth and changes the scale.

The reference derived its bandwidth from the spectrum instead
(:func:`kappa_candidates`, the fourth of ten): 1.95 mm on its own basis, and
1.8 mm on the grid's, well below the grid's mean edge of 5.4 mm on the white
surface, where the kernel barely smooths at all (``K_eff`` 2,437 at 1.95 mm).
"""


def diffusion_kernel(eigenvalues: np.ndarray, eigenvectors: np.ndarray, kappa: float) -> np.ndarray:
    """Riemannian diffusion kernel ``U diag(exp(-kappa^2 Lambda / 2)) U'``.

    Parameters
    ----------
    eigenvalues
        Laplace-Beltrami eigenvalues, ascending, length ``maxk``.
    eigenvectors
        ``(n_vertices, maxk)`` eigenfunctions, one per column. With eigenvectors
        orthonormal under the surface's vertex areas, as
        :func:`laplace_beltrami_basis` gives them, column ``j`` is the heat that
        a unit mass at vertex ``j`` spreads, per unit area.
    kappa
        Bandwidth, in the surface's units (mm on the white surface); the heat
        diffuses for a time ``kappa^2 / 2``. Larger values smooth more.

    Examples
    --------
    >>> import numpy as np
    >>> U = np.eye(3)
    >>> K = diffusion_kernel(np.array([0.0, 1.0, 2.0]), U, kappa=1.0)
    >>> np.allclose(K, np.diag(np.exp(-0.5 * np.array([0.0, 1.0, 2.0]))))
    True
    """
    if not (kappa > 0 and np.isfinite(kappa)):  # written so that NaN fails too
        raise ValueError(f"kappa must be positive and finite, got {kappa}")
    rho = np.exp(-(kappa**2) / 2.0 * _ascending(eigenvalues))
    return _assemble(rho, eigenvectors)


def matern_kernel(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, kappa: float, nu: float = 3.0
) -> np.ndarray:
    """Matern kernel ``U diag((2 nu / kappa^2 + Lambda)^(-nu-1)) U'``, its largest weight 1.

    The weights are formed in log space and divided by the largest, which is
    the diffusion kernel's own scale: a larger ``nu`` or a smaller ``kappa``
    took them below 1e-300 or above 1e300 as written, and the kernel came back
    zero, or ``K A K`` infinite. The scale cancels in the smoothing, which
    normalizes, and the floor of :data:`TRUNCATION_FLOOR` applies relative to it.

    Parameters
    ----------
    kappa
        Bandwidth, in the surface's units (mm on the white surface).
    nu
        Differentiability. The reference script uses 1, 2 or 3.

    Examples
    --------
    >>> import numpy as np
    >>> lam = np.array([0.0, 1.0, 2.0])
    >>> K = matern_kernel(lam, np.eye(3), kappa=1.0, nu=200.0)   # weights near 1e-523 as written
    >>> np.round(np.diag(K), 4).tolist()    # near the diffusion kernel's 0.6065 and 0.3679
    [1.0, 0.6054, 0.367]
    """
    if not (kappa > 0 and np.isfinite(kappa)):
        raise ValueError(f"kappa must be positive and finite, got {kappa}")
    if not (nu > 0 and np.isfinite(nu)):
        raise ValueError(f"nu must be positive and finite, got {nu}")
    lam = _ascending(eigenvalues)
    log_rho = -(nu + 1.0) * np.log(2.0 * nu / kappa**2 + lam)
    return _assemble(np.exp(log_rho - log_rho.max()), eigenvectors)


def _ascending(eigenvalues) -> np.ndarray:
    """The eigenvalues as a flat float64 array, refused unless ascending.

    The order is what the default bandwidth reads: :func:`kappa_candidates`
    takes the highest frequency from the last entry and the 200th from
    position 199. :func:`load_eigenpairs` checks the MATLAB files; arrays
    passed in directly are checked here, where they are consumed.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if not np.all(np.diff(lam) >= 0):
        raise ValueError(
            "eigenvalues must be in ascending order, as load_eigenpairs returns them; "
            "sort them and their eigenvectors together"
        )
    return lam


def _assemble(rho: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """``U diag(rho) U'``, with the truncation floor applied relative to the largest weight."""
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)
    if eigenvectors.ndim != 2:
        raise ValueError(f"eigenvectors must be 2-D, got shape {eigenvectors.shape}")
    if rho.size != eigenvectors.shape[1]:
        raise ValueError(f"{rho.size} eigenvalues for {eigenvectors.shape[1]} eigenvectors")
    largest = float(rho.max()) if rho.size else 0.0
    rho = np.where(rho < TRUNCATION_FLOOR * largest, 0.0, rho)
    return (eigenvectors * rho[np.newaxis, :]) @ eigenvectors.T


def _intrinsic_delaunay(vertices: np.ndarray, faces: np.ndarray):
    """The intrinsic Delaunay triangulation of a mesh, by edge flips.

    A triangle mesh's cotangent Laplacian gives an edge a negative weight
    where the two angles facing it sum past 180 degrees, about one edge in
    thirteen on the ico4 white surface; a heat kernel built on it then dips
    below zero near those edges. Flipping each such edge to the other diagonal
    of its two triangles, with the new diagonal's length taken from the two
    triangles unfolded flat, keeps the surface's metric exactly and changes
    only how it is triangulated; repeated until no edge is left, every weight
    is nonnegative (Bobenko and Springborn, Discrete Comput Geom 38, 2007).

    Returns ``(origin, length)``: for each halfedge ``3 f + k`` of the flipped
    triangulation -- face ``f`` running round its three halfedges in order --
    the vertex it leaves and its length.
    """
    corners = np.asarray(faces, dtype=np.int64)
    points = np.asarray(vertices, dtype=np.float64)
    origin = corners.reshape(-1).copy()
    target = corners[:, [1, 2, 0]].reshape(-1).copy()
    length = np.linalg.norm(points[target] - points[origin], axis=1)
    halfedge = {(int(a), int(b)): h for h, (a, b) in enumerate(zip(origin, target, strict=True))}
    try:
        twin = np.array([halfedge[(int(b), int(a))] for a, b in zip(origin, target, strict=True)])
    except KeyError:
        raise ValueError("the mesh is not closed: an edge has a single triangle") from None

    def following(h):
        return h - h % 3 + (h + 1) % 3

    def preceding(h):
        return h - h % 3 + (h + 2) % 3

    def cot_facing(h):  # cotangent of the angle facing halfedge h, from its triangle's sides
        a, b, c = length[h], length[following(h)], length[preceding(h)]
        s = 0.5 * (a + b + c)
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        return (b * b + c * c - a * a) / (4.0 * area) if area > 0 else -np.inf

    pending = list(range(origin.size))
    while pending:
        h = pending.pop()
        t = twin[h]
        if cot_facing(h) + cot_facing(t) >= -1e-12:
            continue
        # h runs i -> j in triangle (i, j, k), h1 j -> k and h2 k -> i; its twin t
        # runs j -> i in triangle (j, i, l), t1 i -> l and t2 l -> j
        h1, h2, t1, t2 = following(h), preceding(h), following(t), preceding(t)
        k, l_ = origin[h2], origin[t2]
        ij, jk, ki, il, lj = length[h], length[h1], length[h2], length[t1], length[t2]
        # the angles at i on either side of i -> j, added: the angle between i -> k and i -> l
        angle = np.arccos(np.clip((ij * ij + ki * ki - jk * jk) / (2 * ij * ki), -1.0, 1.0))
        angle += np.arccos(np.clip((ij * ij + il * il - lj * lj) / (2 * ij * il), -1.0, 1.0))
        kl = np.sqrt(max(ki * ki + il * il - 2.0 * ki * il * np.cos(angle), 0.0))
        # the two new triangles, (k, i, l) and (l, j, k), take the four outer halfedges
        a, b = h - h % 3, t - t % 3
        moved = {a: h2, a + 1: t1, b: t2, b + 1: h1}
        saved = {slot: (origin[old], length[old], twin[old]) for slot, old in moved.items()}
        for slot, (start, size, partner) in saved.items():
            origin[slot], length[slot], twin[slot] = start, size, partner
            twin[partner] = slot
        origin[a + 2], origin[b + 2] = l_, k
        length[a + 2] = length[b + 2] = kl
        twin[a + 2], twin[b + 2] = b + 2, a + 2
        pending.extend((a, a + 1, b, b + 1))
    every = np.arange(origin.size)
    ends = origin[every - every % 3 + (every + 1) % 3]
    if not (np.array_equal(twin[twin], every) and np.array_equal(origin[twin], ends)):
        raise RuntimeError("the edge flips left the triangulation inconsistent")  # pragma: no cover
    return origin, length


def laplace_beltrami_operator(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    """The cotangent Laplacian of a closed triangle mesh and its lumped vertex areas.

    Built on the mesh's intrinsic Delaunay triangulation (see
    :func:`_intrinsic_delaunay`), so every edge weight is nonnegative and a
    heat kernel built from it is too. Returns ``(stiffness, areas)``: the
    dense ``(n, n)`` matrix ``C`` with ``C 1 = 0``, and each vertex's third of
    the area of the triangles around it, which sum to the surface's area.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([[0, 0, 1], [1, 0, -1], [-1, 1, -1], [-1, -1, -1]], dtype=float)
    >>> f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    >>> C, areas = laplace_beltrami_operator(v, f)
    >>> bool(np.allclose(C.sum(axis=1), 0)), bool((C - np.diag(np.diag(C)) <= 0).all())
    (True, True)
    """
    points = np.asarray(vertices, dtype=np.float64)
    corners = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or corners.ndim != 2 or corners.shape[1] != 3:
        raise ValueError("a mesh is (n, 3) vertices and (m, 3) triangles")
    n = points.shape[0]
    origin, length = _intrinsic_delaunay(points, corners)
    sides = length.reshape(-1, 3)
    half = 0.5 * sides.sum(axis=1)
    area = np.sqrt(
        np.clip(half * (half - sides[:, 0]) * (half - sides[:, 1]) * (half - sides[:, 2]), 0, None)
    )
    if not (area > 0).all():
        raise ValueError(f"{int((area <= 0).sum())} triangle(s) of the mesh have no area")
    following, preceding = sides[:, [1, 2, 0]], sides[:, [2, 0, 1]]
    cotangent = (following**2 + preceding**2 - sides**2) / (4.0 * area[:, None])
    target = origin.reshape(-1, 3)[:, [1, 2, 0]].reshape(-1)
    weight = np.zeros((n, n))
    np.add.at(weight, (origin, target), 0.5 * cotangent.reshape(-1))
    weight += weight.T
    np.fill_diagonal(weight, 0.0)
    stiffness = np.diag(weight.sum(axis=1)) - weight
    areas = np.zeros(n)
    np.add.at(areas, origin, np.repeat(area / 3.0, 3))
    return stiffness, areas


@lru_cache(maxsize=2)
def _grid_basis(side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bundled white surface's Laplace-Beltrami eigenpairs for one hemisphere, read-only."""
    from .surface import load_surface

    part = load_surface("white").hemisphere(side)
    stiffness, areas = laplace_beltrami_operator(part.vertices, part.faces)
    scale = 1.0 / np.sqrt(areas)
    # C phi = lambda M phi with M = diag(areas), as the symmetric problem M^-1/2 C M^-1/2
    eigenvalues, vectors = np.linalg.eigh(stiffness * scale[:, None] * scale[None, :])
    # the constant mode's eigenvalue is 0, and only rounding moves it
    eigenvalues[np.abs(eigenvalues) < 1e-12 * eigenvalues.max()] = 0.0
    eigenvectors = vectors * scale[:, None]
    for array in (eigenvalues, eigenvectors, areas):
        array.setflags(write=False)
    return eigenvalues, eigenvectors, areas


def laplace_beltrami_basis(hemisphere: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grid's Laplace-Beltrami eigenpairs for one hemisphere: what ``rdk`` and ``matern`` use.

    The bundled ico4 white surface's cotangent Laplacian, on its intrinsic
    Delaunay triangulation (:func:`laplace_beltrami_operator`), solved in full:
    ``C phi = lambda M phi`` with ``M`` the diagonal of lumped vertex areas, in
    mm^2. Built once a session, in under half a minute for both hemispheres
    on one core, and kept: 105 MB for both, shared and read-only.

    Parameters
    ----------
    hemisphere
        ``"L"`` or ``"R"``.

    Returns
    -------
    tuple of ndarray
        ``(eigenvalues, eigenvectors, areas)``: the 2,562 eigenvalues,
        ascending, in mm^-2, the first 0; the eigenvectors as columns,
        orthonormal under the areas (``Phi' diag(areas) Phi = I``), the first
        constant; and the areas, which sum to the hemisphere's area.

    Examples
    --------
    >>> import numpy as np
    >>> lam, phi, areas = laplace_beltrami_basis("L")      # doctest: +SKIP
    >>> np.allclose(phi.T @ (areas[:, None] * phi), np.eye(lam.size))  # doctest: +SKIP
    True
    """
    side = str(hemisphere).upper()
    if side not in spec.HEMISPHERE_ORDER:
        raise ValueError(f"hemisphere must be one of {spec.HEMISPHERE_ORDER}, got {hemisphere!r}")
    return _grid_basis(side)


NUM_HARMONICS = 33
"""Terms ``concon`` sums, so degrees ``l = 0 .. 32``.

The count is hardcoded and ``--OPT_VAL_num_harm`` is **ignored**: ``c3_main``
prints "num_harmonics now global constant (for speed)" at startup, and passing
it 9, 17, 25, 33, 49 or 65 gives a bit-identical kernel. The pipeline's
``--OPT_VAL_num_harm 33`` therefore changes nothing, though it happens to name
the value actually used.

Fixed at 33 by measurement rather than by reading: across bandwidths 0.00125
to 0.02, 33 terms reproduces the binary's peak to 0.06-0.1% and its shape to
rms 0.0002, where 32 terms drifts to 4% at the narrowest bandwidth and 34
terms overshoots. The truncation matters -- at sigma 0.005 the last term still
carries 0.28% of the peak, 270.267 -- so a port that summed further would stop
reproducing the released cohorts.
"""

DEFAULT_SIGMA = 0.005
"""Bandwidth the released cohorts use, from ``--sigma 0.005``."""

FINAL_THRESHOLD = 1e-9
"""``--final_thold 0.000000001``: after dividing by the streamline count,
``compute_kernel.cpp`` writes a pair only ``if(temp > final_thold)``. Applied
by :func:`smooth` on the same scale, before the unit-mass normalization."""

KERNEL_EPSILON = 0.001
"""Where ``concon`` cuts the kernel off, from ``--epsilon``.

``compute_kernel.cpp`` scans ``x = cos(theta)`` down from 1.0 in steps of
1e-4 and stops at the first ``x`` where the product ``K(x) K(1)`` falls below
this; vertices beyond that angle get nothing. The truncated series plunges
through zero there, so anything from 1e-6 to 0.1 gives the same cutoff to
three decimals, which is why sweeping the flag against the binary appeared to
do nothing.
"""

DENSE_BLOCK = 4096
"""Endpoints per block on the dense path: a ``(5124, 4096)`` kernel block is 168 MB."""

SPARSE_BLOCK = 65536
"""Endpoints per block on the sparse path.

Each block ends in a ``5124 x 5124`` densification, so small blocks cost far
more memory traffic than they save: 4096 meant 245 such passes per million
endpoints, 65536 means 16.
"""

SPARSE_PAIR_BUDGET = 10**9
"""Bytes a block's (vertex, endpoint) pairs may take at the sparse path's peak, a side at a time.

An endpoint reaches the vertices of its hemisphere inside the cutoff: 29 at the
default bandwidth, 1,911 at sigma 0.5. At :data:`PAIR_BYTES` a pair, a block of
:data:`SPARSE_BLOCK` endpoints takes 120 MB at the default and 8 GB at sigma
0.5, so the default block is shrunk to stay within this (:func:`_sparse_block`).
Two dense ``5124 x 5124`` arrays, 420 MB, sit beside the pairs whatever the block.
"""

PAIR_BYTES = 64
"""Bytes one pair costs at the sparse path's peak.

``cKDTree.sparse_distance_matrix`` returns a pair as two indices and a distance,
24 bytes; the block's other arrays -- the rows and columns, the vertices and
points they gather, the cosines and the lookup's temporaries -- grow with the
pairs as well, to about 64 bytes a pair measured with ``tracemalloc``
(PORTING.md item 41).
"""


def _check_sigma(sigma) -> None:
    """Refuse a bandwidth that is not finite and above 0, which the series took without a word.

    A NaN made a cutoff of 180 degrees and a kernel of NaNs, and a negative
    sigma weights the high degrees up rather than down.
    """
    if not (sigma > 0 and np.isfinite(sigma)):  # written so that NaN fails too
        raise ValueError(f"sigma must be positive and finite, got {sigma}")


@lru_cache(maxsize=32)
def kernel_cutoff(
    sigma: float, epsilon: float = KERNEL_EPSILON, harmonics: int = NUM_HARMONICS
) -> float:
    """Angle in radians beyond which the kernel is zero.

    Reproduces ``compute_kernel.cpp``'s scan exactly -- the product
    ``K(x) K(1)`` against ``epsilon``, in steps of 1e-4 in the cosine -- on the
    exact series rather than the binary's tables, so the cutoff lands on the
    same grid vertices as :attr:`KernelTable.cutoff` wherever the two agree.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(np.degrees(kernel_cutoff(0.005))), 3)
    12.231
    >>> round(float(np.degrees(kernel_cutoff(0.01))), 3)
    17.369
    """
    _check_sigma(sigma)
    cosines = np.arange(1.0, -1.0, -0.0001)
    values = _legendre_series(cosines, sigma, harmonics)
    below = np.nonzero(values * values[0] < epsilon)[0]
    return float(np.arccos(np.clip(cosines[below[0]], -1.0, 1.0))) if below.size else np.pi


def _legendre_series(cosine: np.ndarray, sigma: float, harmonics: int) -> np.ndarray:
    """The untruncated sum, shared by the closed-form kernel and its cutoff scan.

    The recurrence runs in place. With ``quantized=False`` this runs over a
    ``(5124, 4096)`` block per batch of endpoints, and allocating a fresh array
    per term would cost 168 MB each time; reusing three buffers made that path
    about three times faster end to end. The default path reads the tables.
    """
    degree = np.arange(harmonics)
    weight = (2 * degree + 1) ** 1.5 / np.sqrt(4 * np.pi) * np.exp(-degree * (degree + 1) * sigma)

    # Work on a flat view: a scalar argument becomes a 0-d array, which numpy
    # refuses as an `out=` target, and the recurrence writes through `out`.
    shape = np.shape(cosine)
    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)

    previous = np.ones_like(cosine)
    total = weight[0] * previous
    if harmonics == 1:
        return total.reshape(shape)

    current = cosine.copy()
    term = np.empty_like(cosine)
    np.multiply(current, weight[1], out=term)
    total += term

    scratch = np.empty_like(cosine)
    for order in range(1, harmonics - 1):
        # scratch <- ((2k+1) cos * current - k * previous) / (k + 1)
        np.multiply(cosine, current, out=scratch)
        scratch *= 2 * order + 1
        np.multiply(previous, float(order), out=term)
        scratch -= term
        scratch /= order + 1
        previous, current, scratch = current, scratch, previous
        np.multiply(current, weight[order + 1], out=term)
        total += term
    return total.reshape(shape)


HARMONIC_SAMPLES_EXPONENT = 5
"""``--OPT_VAL_exp_num_harm_samps 5``: ``concon`` tabulates each spherical
harmonic at ``10**5`` cosine samples per unit (``subject.cpp``:
``num_harm_samps = pow(10, exp_num_harm_samps)``)."""

KERNEL_SAMPLES_EXPONENT = 6
"""``--OPT_VAL_exp_num_kern_samps 6``: the kernel itself is tabulated at
``10**6`` cosine samples per unit and read back by integer truncation."""


@dataclass(frozen=True)
class KernelTable:
    """``concon``'s kernel exactly as ``c3_main`` evaluates it: a lookup table.

    ``compute_kernel.cpp`` never evaluates the series at a point. It reads
    ``kern_lookup_table[(int)(dot * M + M)]``, a table of ``2M + 1`` values over
    the cosine in ``[-1, 1]``, and that table was itself filled from a harmonic
    table read the same way, ``harm_lookup_table[l][(int)(x * L + L)]``. Both
    reads truncate, so every lookup lands on the sample at or below the true
    cosine -- a slightly larger angle, a slightly smaller kernel -- and the
    bias grows with ``l(l+1)``. That is the amplitude offset the exact series
    leaves against the binary, 0.14% on the tutorial subject, and
    reproducing the tables removes it: on a single streamline the binary's peak entry is
    ``K(1.0) * K(1 - 1e-12) = 270.2668 * 269.9403``, whose square root is the
    measured 270.1035, at all five bandwidths tested.

    Attributes
    ----------
    values
        The ``2M + 1`` tabulated kernel values.
    samples
        ``M``.
    cutoff_cosine
        Vertices with ``dot < cutoff_cosine`` receive nothing, found by the
        binary's own scan: ``x`` from 1.0 down in steps of 1e-4 until
        ``K(x) * K(1) < epsilon``.
    """

    values: np.ndarray
    samples: int
    cutoff_cosine: float

    @property
    def cutoff(self) -> float:
        """Cutoff angle in radians."""
        return float(np.arccos(np.clip(self.cutoff_cosine, -1.0, 1.0)))

    def __call__(self, cosine, truncate: bool = True) -> np.ndarray:
        """Kernel values by truncation lookup, zero beyond the cutoff unless ``truncate`` is off.

        A NaN cosine gives NaN: the cast to an index mapped it onto an entry of
        the table, so a NaN position read as a finite kernel value.
        """
        # No `out=` anywhere here: a scalar argument is a 0-d array, which
        # numpy refuses as an output buffer.
        cosine = np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)
        missing = np.isnan(cosine)
        nan = bool(missing.any())
        if nan:
            cosine = np.where(missing, 1.0, cosine)  # any entry, so the cast below stays defined
        index = np.clip(
            (cosine * self.samples + self.samples).astype(np.int64), 0, 2 * self.samples
        )
        values = self.values[index]
        if nan:
            values = np.where(missing, np.nan, values)
        if not truncate:
            return values
        return np.where(cosine < self.cutoff_cosine, 0.0, values)


@lru_cache(maxsize=8)
def concon_kernel_table(
    sigma: float,
    harmonics: int = NUM_HARMONICS,
    epsilon: float = KERNEL_EPSILON,
    harmonic_exponent: int = HARMONIC_SAMPLES_EXPONENT,
    kernel_exponent: int = KERNEL_SAMPLES_EXPONENT,
) -> KernelTable:
    """Build the two lookup tables ``c3_main`` builds, and the cutoff it scans.

    Examples
    --------
    >>> table = concon_kernel_table(0.005)
    >>> round(float(np.sqrt(table(1.0) * table(1.0 - 1e-12))), 4)   # the binary's peak entry
    270.1035
    >>> round(float(np.degrees(table.cutoff)), 3)
    12.231
    """
    _check_sigma(sigma)
    degree = np.arange(harmonics)
    big_l = 10**harmonic_exponent
    big_m = 10**kernel_exponent

    # harm_lookup_table[l][j + L] = EvalSH(l, 0, 0, acos(j / L)): the normalized
    # harmonic Y_l^0, which carries sqrt((2l+1)/4pi).
    x_h = np.arange(-big_l, big_l + 1, dtype=np.float64) / big_l
    harmonic = (
        _legendre_polynomials(x_h, harmonics) * np.sqrt((2 * degree + 1) / (4 * np.pi))[:, None]
    )

    # kern_lookup_table[i + M] = sum_l exp(-sigma l(l+1)) (2l+1) harm[l][(int)(x_i L + L)]
    x_k = np.arange(-big_m, big_m + 1, dtype=np.float64) / big_m
    index = (x_k * big_l + big_l).astype(np.int64)
    np.clip(index, 0, 2 * big_l, out=index)
    weight = np.exp(-sigma * degree * (degree + 1)) * (2.0 * degree + 1.0)
    # Sum on the 200,001-point harmonic table, then gather once: every entry of
    # the kernel table is the same sum in the same order, so this is bit for
    # bit the binary's loop over the 2,000,001 entries at a tenth of the work.
    summed = np.zeros(x_h.size, dtype=np.float64)
    for order in range(harmonics):
        summed += weight[order] * harmonic[order]
    values = summed[index]
    np.maximum(values, 1e-20, out=values)  # fmax(ret_val, 1e-20)

    # The cutoff scan in compute_kernel.cpp, on the table.
    scan = 1.0 - 1e-4 * np.arange(0, int(2.0 / 1e-4) + 1)
    scan_index = np.clip((scan * big_m + big_m).astype(np.int64), 0, 2 * big_m)
    product = values[scan_index] * values[2 * big_m]
    below = np.nonzero(product < epsilon)[0]
    cutoff_cosine = float(scan[below[0]]) if below.size else -1.0
    return KernelTable(values=values, samples=big_m, cutoff_cosine=cutoff_cosine)


def _legendre_with_derivative(x, degree: int):
    """Yield ``(l, P_l(x), dP_l/dx)`` for ``l = 0..degree`` by the standard recurrences.

    Shared with ConSEAL's kernel builder, which needs the derivative as well.
    """
    x = np.asarray(x, dtype=np.float64)
    p_prev, p_cur = np.ones_like(x), x.copy()
    d_prev, d_cur = np.zeros_like(x), np.ones_like(x)
    yield 0, p_prev, d_prev
    if degree >= 1:
        yield 1, p_cur, d_cur
    for h in range(1, degree):
        p_next = ((2 * h + 1) * x * p_cur - h * p_prev) / (h + 1)
        d_next = ((2 * h + 1) * (p_cur + x * d_cur) - h * d_prev) / (h + 1)
        yield h + 1, p_next, d_next
        p_prev, p_cur, d_prev, d_cur = p_cur, p_next, d_cur, d_next


def _legendre_polynomials(x: np.ndarray, terms: int) -> np.ndarray:
    """``P_l(x)`` for ``l = 0 .. terms-1`` as rows."""
    out = np.empty((terms, x.size), dtype=np.float64)
    for order, p, _ in _legendre_with_derivative(x, terms - 1):
        out[order] = p
    return out


def spherical_heat_kernel(
    cosine,
    sigma: float,
    harmonics: int = NUM_HARMONICS,
    epsilon: float | None = KERNEL_EPSILON,
    quantized: bool = True,
):
    r"""The kernel ``concon`` applies, on the unit sphere.

    .. math::
        K_\sigma(\cos\gamma) = \sum_{l=0}^{32}
            \frac{(2l+1)^{3/2}}{\sqrt{4\pi}}\, e^{-l(l+1)\sigma}\, P_l(\cos\gamma)

    zero beyond the cutoff, and -- by default -- read through the same two
    truncation-indexed lookup tables ``c3_main`` reads it through
    (:class:`KernelTable`), which is what the released cohorts carry.

    **This is not the heat kernel**, whose weight is :math:`(2l+1)/4\pi`.
    ``sigma_opt.cpp`` sums ``exp(-sigma*l*(l+1)) * (2l+1) * harm_lookup[l]``,
    and ``subject.cpp`` fills that lookup with ``sh::EvalSH(l, 0, 0, acos(x))``
    -- the *normalized* spherical harmonic :math:`Y_l^0`, which already carries
    :math:`\sqrt{(2l+1)/4\pi}`. The two factors compound into
    :math:`(2l+1)^{3/2}`, which looks like a Legendre polynomial was intended
    where a normalized harmonic was used. Intended or not, it is what produced
    every released cohort, so it is what this package reproduces. The extra
    :math:`\sqrt{2l+1}` upweights high degrees and makes the kernel markedly
    narrower than a true heat kernel.

    Parameters
    ----------
    cosine
        Cosines of the angle between points, any shape. Clipped to ``[-1, 1]``.
    sigma
        Bandwidth, positive and finite: a diffusion time on the unit sphere.
        The released cohorts use 0.005, where the kernel falls to half height
        6.0 degrees from its centre and to zero at 12.2 degrees, about 7 and
        15 mm on the white surface, which averages 1.2 mm a degree. Below
        about 0.001 the 33 terms, not sigma, set the width: half height stays
        at 3.6 to 4.0 degrees (3.96 at 0.001, 3.62 at 1e-5), so a smaller
        sigma no longer makes the kernel narrower, and :func:`smooth` warns.
    harmonics
        Number of terms. ``concon`` hardcodes 33 -- see :data:`NUM_HARMONICS`.
    epsilon
        Cutoff threshold. ``None`` leaves the series untruncated in angle and
        evaluates it in closed form, which is useful for studying it but is not
        what the pipeline did: the binary's tables clamp the negative lobes
        beyond the cutoff to 1e-20, so past it they hold a floor, not the kernel.
    quantized
        ``True`` reproduces the binary's lookup tables exactly and is the
        default. ``False`` evaluates the series in closed form: the kernel the
        tables approximate, about 0.06% higher at the peak. Ignored when
        ``epsilon`` is ``None``.

    Notes
    -----
    Verified against ``c3_main`` itself, run on a single streamline so its
    output is the kernel: the quantized form reproduces the binary's peak entry
    to six digits at five bandwidths, and its ring values at sigma 0.005 to rms
    0.0005; against the binary run on the tutorial subject, at full scale, the
    density correlates at 1.000000 once the grid's south pole, which the binary
    never writes, is emptied. ``tests/reference/concon_probe.py`` and
    ``concon_fullscale.py`` regenerate the measurements.

    Examples
    --------
    >>> import numpy as np
    >>> from sbci.smoothing import spherical_heat_kernel
    >>> centre = float(spherical_heat_kernel(1.0, 0.005))
    >>> round(centre, 3)
    270.267
    >>> float(spherical_heat_kernel(np.cos(np.radians(20.0)), 0.005))  # past the cutoff
    0.0
    """
    _check_sigma(sigma)
    cosine = np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)
    if epsilon is None:
        return _legendre_series(cosine, sigma, harmonics)
    if quantized:
        return concon_kernel_table(sigma, harmonics, epsilon)(cosine)
    total = _legendre_series(cosine, sigma, harmonics)
    return np.where(cosine < np.cos(kernel_cutoff(sigma, epsilon, harmonics)), 0.0, total)


def _sparse_block(cutoff: float, n_hemisphere: int) -> int:
    """Endpoints a block of the sparse path takes: :data:`SPARSE_BLOCK`, fewer for a wide kernel.

    An endpoint pairs with the vertices of its hemisphere inside the cutoff,
    the cap's share of them, ``n_hemisphere * (1 - cos(cutoff)) / 2``: 29 of
    2562 at the default bandwidth, every one at sigma 1. The block holds its
    pairs within :data:`SPARSE_PAIR_BUDGET`, at :data:`PAIR_BYTES` each.

    Examples
    --------
    >>> _sparse_block(concon_kernel_table(DEFAULT_SIGMA).cutoff, 2562) == SPARSE_BLOCK
    True
    >>> _sparse_block(np.pi, 2562)    # the whole hemisphere: 2562 pairs an endpoint
    6098
    """
    pairs = max(1.0, n_hemisphere * (1.0 - np.cos(cutoff)) / 2.0)
    return int(max(1, min(SPARSE_BLOCK, SPARSE_PAIR_BUDGET // (PAIR_BYTES * pairs))))


def endpoint_density(
    vertices,
    points_in,
    points_out,
    hemisphere_in,
    hemisphere_out,
    vertex_hemisphere,
    sigma: float,
    harmonics: int = NUM_HARMONICS,
    normalize: str | None = None,
    block: int | None = None,
    method: str = "sparse",
    progress=None,
    quantized: bool = True,
):
    """Smooth streamline endpoints into a density with the spherical kernel.

    Each endpoint contributes a kernel centred on its **continuous** position,
    not on the nearest grid vertex, which is what ``c3_main`` does. Endpoints
    only contribute to vertices of their own hemisphere.

    Parameters
    ----------
    vertices
        ``(n, 3)`` grid vertices on the unit sphere.
    points_in, points_out
        ``(s, 3)`` continuous endpoint positions on the unit sphere.
    hemisphere_in, hemisphere_out, vertex_hemisphere
        Which hemisphere each endpoint and each vertex belongs to.
    sigma, harmonics, quantized
        Passed to :func:`spherical_heat_kernel`.
    normalize
        ``None``, the default, leaves each endpoint's kernel unscaled, which is
        what ``c3_main`` does -- it divides the finished density by the
        streamline count instead. ``"sum"`` scales each kernel to sum to one
        over the grid; it was a workaround for a scale mismatch that turned out
        to be the kernel's weight, and is kept only for comparison.
    block
        Endpoints processed at a time, a positive whole number; defaults to
        :data:`DENSE_BLOCK` on the dense path, and on the sparse path to
        :data:`SPARSE_BLOCK`, or fewer where the kernel reaches so many
        vertices that a block's pairs would pass :data:`SPARSE_PAIR_BUDGET`.
    method
        ``"sparse"``, the default, finds each endpoint's neighbourhood with a
        KD-tree and evaluates the kernel only there: the kernel is zero past
        about 12 degrees, so only ~31 of the 5124 vertices matter per endpoint
        and the dense evaluation was 99.4% zeros. ``"dense"`` evaluates every
        vertex and is kept as the check the sparse path is tested against.
    progress
        Optional callable ``progress(done, total)`` invoked after each block.

    Returns
    -------
    The symmetric ``(n, n)`` density.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    points_in = np.asarray(points_in, dtype=np.float64)
    points_out = np.asarray(points_out, dtype=np.float64)
    if points_in.shape != points_out.shape:
        raise ValueError(f"endpoints disagree: {points_in.shape} in, {points_out.shape} out")
    if normalize not in ("sum", None):
        raise ValueError(f"normalize must be 'sum' or None, got {normalize!r}")
    if method not in ("sparse", "dense"):
        raise ValueError(f"method must be 'sparse' or 'dense', got {method!r}")

    vertex_hemisphere = np.asarray(vertex_hemisphere)
    hemisphere_in = np.asarray(hemisphere_in)
    hemisphere_out = np.asarray(hemisphere_out)
    n = vertices.shape[0]
    total = points_in.shape[0]
    _check_sigma(sigma)
    sized = block is None  # the sparse path fits its default block to the kernel's reach, below
    if block is None:
        block = DENSE_BLOCK if method == "dense" else SPARSE_BLOCK
    # a block of -1 made an empty range and a density of zeros
    if isinstance(block, (bool, np.bool_)) or not isinstance(block, (int, np.integer)) or block < 1:
        raise ValueError(f"block must be a positive whole number of endpoints, got {block!r}")
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or points_in.ndim != 2
        or points_in.shape[1] != 3
    ):
        raise ValueError("vertices and endpoints must be (n, 3) unit vectors")
    if hemisphere_in.shape != (total,) or hemisphere_out.shape != (total,):
        raise ValueError(f"hemisphere flags must have one entry per streamline ({total})")
    if vertex_hemisphere.shape != (n,):
        raise ValueError(f"vertex_hemisphere has {vertex_hemisphere.size} entries for {n} vertices")
    for name, array in (
        ("vertices", vertices),
        ("points_in", points_in),
        ("points_out", points_out),
    ):
        # Written so that a NaN fails the check too
        if array.size and not (np.abs(np.linalg.norm(array, axis=1) - 1.0) <= 1e-6).all():
            raise ValueError(f"{name} must be finite unit vectors")
    flags = np.unique(np.concatenate([hemisphere_in, hemisphere_out]))
    unknown = np.setdiff1d(flags, np.unique(vertex_hemisphere))
    if unknown.size:
        raise ValueError(
            f"endpoint hemisphere flags {unknown.tolist()} match no vertex hemisphere; an "
            "endpoint with no vertices to land on would otherwise be dropped silently"
        )

    def kernel(cosine):
        return spherical_heat_kernel(cosine, sigma, harmonics, quantized=quantized)

    if method == "dense":

        def columns(points, hemisphere):
            values = np.asarray(kernel(vertices @ points.T), dtype=np.float64)
            values[vertex_hemisphere[:, None] != hemisphere[None, :]] = 0.0
            np.maximum(values, 0.0, out=values)
            if normalize == "sum":
                colsum = values.sum(axis=0)
                colsum[colsum == 0] = 1.0
                values = values / colsum
            return values

        density = np.zeros((n, n), dtype=np.float64)
        for start in range(0, total, block):
            stop = min(start + block, total)
            density += (
                columns(points_in[start:stop], hemisphere_in[start:stop])
                @ columns(points_out[start:stop], hemisphere_out[start:stop]).T
            )
            if progress is not None:
                progress(stop, total)
        return density + density.T

    from scipy import sparse
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    cutoff = (
        concon_kernel_table(sigma, harmonics, KERNEL_EPSILON).cutoff
        if quantized
        else kernel_cutoff(sigma, KERNEL_EPSILON, harmonics)
    )
    # Chord length of the cutoff angle, with slack: the kernel itself decides
    # membership (it is zero past the cutoff), the tree only has to over-cover.
    radius = float(np.sqrt(max(0.0, 2.0 - 2.0 * np.cos(cutoff)))) * (1.0 + 1e-6) + 1e-9
    sides = np.unique(vertex_hemisphere)
    trees = {int(h): cKDTree(vertices[vertex_hemisphere == h]) for h in sides}
    members = {int(h): np.nonzero(vertex_hemisphere == h)[0] for h in sides}
    if sized:
        block = _sparse_block(cutoff, max((part.size for part in members.values()), default=0))

    def sparse_columns(points, hemisphere):
        rows, cols, vals = [], [], []
        for h in sides:
            chosen = np.nonzero(hemisphere == h)[0]
            if chosen.size == 0:
                continue
            # Every (vertex, endpoint) pair within the cutoff, found tree against
            # tree in C rather than as a Python list per endpoint.
            pairs = trees[int(h)].sparse_distance_matrix(
                cKDTree(points[chosen]), radius, output_type="ndarray"
            )
            if pairs.size == 0:
                continue
            row = members[int(h)][pairs["i"]]
            col = chosen[pairs["j"]]
            cosine = np.einsum("ij,ij->i", vertices[row], points[col])
            value = np.asarray(kernel(cosine), dtype=np.float64)
            keep = value > 0.0
            rows.append(row[keep])
            cols.append(col[keep])
            vals.append(value[keep])
        if not rows:
            return csr_matrix((n, points.shape[0]), dtype=np.float64)
        matrix = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, points.shape[0]),
        )
        if normalize == "sum":
            colsum = np.asarray(matrix.sum(axis=0)).ravel()
            colsum[colsum == 0] = 1.0
            matrix = (matrix @ sparse.diags(1.0 / colsum)).tocsr()
        return matrix

    density = np.zeros((n, n), dtype=np.float64)
    for start in range(0, total, block):
        stop = min(start + block, total)
        product = (
            sparse_columns(points_in[start:stop], hemisphere_in[start:stop])
            @ sparse_columns(points_out[start:stop], hemisphere_out[start:stop]).T
        )
        density += product.toarray()
        if progress is not None:
            progress(stop, total)
    return density + density.T


def endpoint_positions(endpoints, surface=None):
    """Continuous unit-sphere positions of each endpoint, as ``concon`` sees them.

    ``concon`` centres a kernel on where the streamline actually crossed the
    surface, not on the nearest vertex, so the file stores each crossing as a
    triangle plus barycentric weights. This rebuilds the position from them.

    A position lies within one edge of the grid of the endpoint's stored
    vertex when its triangle index refers to the grid's own face list
    (SPEC_QUESTIONS.md item 13). Endpoints whose positions lie farther away
    are refused: their triangles name other triangles than were crossed.

    Parameters
    ----------
    endpoints
        :class:`Endpoints` carrying barycentric coordinates.
    surface
        The spherical grid mesh; loaded from the bundled surfaces if omitted.

    Returns
    -------
    tuple of ndarray
        ``(points_in, points_out)``, each ``(s, 3)`` on the unit sphere.
    """
    points_in, points_out, offsets, edge = _vertex_offsets(endpoints, surface)
    far = offsets > edge
    if far.any():
        raise ValueError(
            f"{int(far.sum()):,} of {offsets.size:,} endpoint positions lie farther than one "
            f"edge of the grid ({edge:.2f} degrees) from their stored vertex, up to "
            f"{offsets.max():.1f} degrees: their triangle indices name other triangles than "
            "the grid's own face list does (SPEC_QUESTIONS.md item 13)"
        )
    return points_in, points_out


def _vertex_offsets(endpoints, surface=None):
    """The positions rebuilt from the triangles, and how far each lies from its stored vertex.

    Returns ``(points_in, points_out, offsets, edge)``: the offsets in degrees,
    the ``in`` ends' and then the ``out`` ends', and the grid's longest edge in
    degrees, as far as a position may lie from its vertex. ``sbci validate``
    reports them, and :func:`endpoint_positions` refuses an offset past the edge.
    """
    if not endpoints.has_positions:
        raise MissingDataError(_NO_POSITIONS)
    if surface is None:
        from .surface import load_surface

        surface = load_surface("sphere")
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    # Triangle indices are stored per hemisphere, so they need the same offset
    # the vertex indices get. Take it from the mesh rather than a constant, so
    # it stays right if the grid ever changes.
    faces_per_hemi = faces.shape[0] // 2

    def rebuild(triangle, bary, surf):
        local = np.asarray(triangle, dtype=np.int64)
        if local.size and (local.min() < 0 or local.max() >= faces_per_hemi):
            raise ValueError(
                f"triangle indices must lie in 0..{faces_per_hemi - 1} within a hemisphere; "
                f"got {local.min()}..{local.max()}"
            )
        index = local + Endpoints.global_hemisphere_offset(surf, faces_per_hemi)
        corners = vertices[faces[index]]  # (s, 3, 3)
        point = np.einsum("sk,skj->sj", np.asarray(bary, dtype=np.float64), corners)
        return point / np.linalg.norm(point, axis=1, keepdims=True)

    points_in = rebuild(endpoints.tri_in, endpoints.bary_in, endpoints.surf_in)
    points_out = rebuild(endpoints.tri_out, endpoints.bary_out, endpoints.surf_out)
    if 2 * endpoints.n_per_hemi != vertices.shape[0]:
        raise ValueError(
            f"the endpoints are on a grid of {2 * endpoints.n_per_hemi} vertices and the "
            f"surface has {vertices.shape[0]}, so their vertex indices name other vertices there"
        )
    unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    cosine = np.concatenate(
        [
            np.einsum("ij,ij->i", points_in, unit[endpoints.global_vertex_in]),
            np.einsum("ij,ij->i", points_out, unit[endpoints.global_vertex_out]),
        ]
    )
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    longest = np.einsum("ij,ij->i", unit[edges[:, 0]], unit[edges[:, 1]]).min()  # its cosine
    return (
        points_in,
        points_out,
        np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))),
        float(np.degrees(np.arccos(np.clip(longest, -1.0, 1.0)))),
    )


def read_concon_endpoints(path):
    """Read ``subject_xing_sphere_avg_coords.tsv``, the input ``c3_main`` takes.

    Returns ``(points_in, points_out, hemisphere_in, hemisphere_out)``, with
    hemispheres as 0 for left and 1 for right. The file stores ``1 - surface``
    in its hemisphere columns, which is undone here.
    """
    raw = np.loadtxt(path, skiprows=1)
    if raw.ndim == 1:
        raw = raw[None]
    if raw.shape[1] != 10:
        raise ValueError(f"{path}: expected 10 columns, got {raw.shape[1]}")

    def unit(block):
        return block / np.linalg.norm(block, axis=1, keepdims=True)

    return (
        unit(raw[:, 2:5]),
        unit(raw[:, 7:10]),
        (1 - raw[:, 1]).astype(np.int64),
        (1 - raw[:, 6]).astype(np.int64),
    )


def kappa_candidates(eigenvalues: np.ndarray, n: int = 10, index: int = 200) -> np.ndarray:
    """Log-spaced bandwidths spanning the usable range of the spectrum, the reference's way.

    The low end is the bandwidth at which the *highest* represented frequency
    is still damped only to 0.9, and the high end the bandwidth at which the
    ``index``-th eigenvalue is damped to 0.001. Below the low end the kernel
    barely smooths; above the high end it has thrown away everything but the
    coarsest modes. ``rdk_smoothed_concon_compute.m`` smooths at the fourth of
    ten; :func:`smooth` no longer does, since the highest frequency is the
    mesh's rather than the cortex's (:data:`DEFAULT_KAPPA`).

    Examples
    --------
    >>> lam = np.linspace(0.0, 0.5, 300)
    >>> candidates = kappa_candidates(lam)
    >>> candidates.size, bool(np.all(np.diff(candidates) > 0))
    (10, True)
    """
    lam = _ascending(eigenvalues)
    if lam.size < index:
        raise ValueError(f"need at least {index} eigenvalues, got {lam.size}")
    kappa_min = np.sqrt(-2.0 * np.log(0.9) / lam[-1])
    kappa_max = np.sqrt(-2.0 * np.log(0.001) / lam[index - 1])
    return np.exp(np.linspace(np.log(kappa_min), np.log(kappa_max), n))


def _indices(name, values) -> np.ndarray:
    """Whole-number indices as ``int64``, checked before the cast, which truncates 2.5 to 2."""
    raw = np.asarray(values).ravel()
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} holds {raw.dtype} values, not indices")
    if raw.dtype.kind == "f" and raw.size and not np.all(raw == np.round(raw)):
        raise ValueError(f"{name} holds indices, which are whole numbers")
    return raw.astype(np.int64)


BARYCENTRIC_TOLERANCE = 1e-4
"""How far a row of barycentric weights may sum from one: the files keep them in
float32, which moves a sum by about 1e-7."""


def _weights(name, values) -> np.ndarray:
    """Barycentric weights as ``float64``, checked before the cast and after it.

    NumPy reads text such as ``b"0.5"`` as a number when cast, and a compound
    value raised a ``TypeError``; a NaN weight loaded, validated, and made
    ``smooth()`` fail on a NaN position. Each row has to sum to one, within
    :data:`BARYCENTRIC_TOLERANCE`: a row of zeros put its endpoint nowhere, and
    another sum weighed its streamline by that sum in ConSEAL's counts. A
    weight may be slightly negative, as the point search leaves one for a
    point just outside its triangle.
    """
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} holds {raw.dtype} values, not barycentric weights")
    weights = raw.astype(np.float64)
    bad = int(np.count_nonzero(~np.isfinite(weights)))
    if bad:
        raise ValueError(
            f"{name} holds {bad} weight(s) that are not finite, which would put their "
            "endpoints nowhere and make the density NaN"
        )
    if weights.ndim == 2 and weights.shape[1] == 3:  # another shape is refused with the count
        sums = weights.sum(axis=1)
        off = np.abs(sums - 1.0) > BARYCENTRIC_TOLERANCE
        if off.any():
            raise ValueError(
                f"{name} holds {int(off.sum())} row(s) that do not sum to 1 (the first sums "
                f"to {sums[off][0]:.6g}); barycentric weights sum to 1, within "
                f"{BARYCENTRIC_TOLERANCE:g}, and a row of zeros puts its endpoint nowhere"
            )
    return weights


@dataclass
class Endpoints:
    """Streamline endpoints on the two hemispheres.

    Attributes
    ----------
    surf_in, surf_out
        Hemisphere of each endpoint, ``0`` for left and ``1`` for right.
    vtx_in, vtx_out
        Zero-based vertex index within that hemisphere.
    tri_in, tri_out
        Optional zero-based triangle index within that hemisphere, and
        ``bary_in``/``bary_out`` the barycentric weights inside it. Together
        they give the endpoint's continuous position on the surface, which
        matters for a kernel evaluated off-grid: snapping to the nearest vertex
        moves an endpoint 1.5 degrees on average, against kernels a few degrees
        wide. Without them an endpoint is known only to the nearest vertex.
    n_per_hemi
        Vertices per hemisphere, 2562 on ico4.
    """

    surf_in: np.ndarray
    surf_out: np.ndarray
    vtx_in: np.ndarray
    vtx_out: np.ndarray
    n_per_hemi: int = spec.N_VERTICES_PER_HEMI
    tri_in: np.ndarray | None = None
    tri_out: np.ndarray | None = None
    bary_in: np.ndarray | None = None
    bary_out: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Coerce the fields to arrays and check they describe the same streamlines.

        Indices have to be whole numbers on the grid, and barycentric weights
        finite numbers whose rows sum to one; each is checked before it is cast.
        """
        for name in ("surf_in", "surf_out"):
            raw = np.asarray(getattr(self, name)).ravel()
            # checked before the cast: int8 wraps 256 to 0 and truncates 1.9 to 1
            if raw.size and not np.isin(raw, (0, 1)).all():
                raise ValueError(f"{name} must be 0 (left) or 1 (right)")
            setattr(self, name, raw.astype(np.int8))
        for name in ("vtx_in", "vtx_out"):
            setattr(self, name, _indices(name, getattr(self, name)))
        count = self.vtx_in.size
        for name in ("surf_in", "surf_out", "vtx_out"):
            if getattr(self, name).size != count:
                raise ValueError(
                    f"{name} has {getattr(self, name).size} entries for {count} streamlines"
                )
        for name in ("surf_in", "surf_out"):
            flags = getattr(self, name)
            if flags.size and not np.isin(flags, (0, 1)).all():
                raise ValueError(f"{name} must be 0 (left) or 1 (right)")
        # Within the hemisphere: an index past it is stored as a whole-grid index that
        # reads back as a vertex of the other hemisphere (save, then load).
        for name in ("vtx_in", "vtx_out"):
            vertex = getattr(self, name)
            if vertex.size and (vertex.min() < 0 or vertex.max() >= self.n_per_hemi):
                raise ValueError(
                    f"{name} out of range for {self.n_per_hemi} vertices per hemisphere: "
                    f"[{vertex.min()}, {vertex.max()}]. One-based indices need converting."
                )
        optional = (self.tri_in, self.tri_out, self.bary_in, self.bary_out)
        if any(item is not None for item in optional):
            if any(item is None for item in optional):
                raise ValueError("positions need all of tri_in, tri_out, bary_in and bary_out")
            self.tri_in = _indices("tri_in", self.tri_in)
            self.tri_out = _indices("tri_out", self.tri_out)
            self.bary_in = _weights("bary_in", self.bary_in)
            self.bary_out = _weights("bary_out", self.bary_out)
            for name, expected in (
                ("tri_in", (count,)),
                ("tri_out", (count,)),
                ("bary_in", (count, 3)),
                ("bary_out", (count, 3)),
            ):
                if getattr(self, name).shape != expected:
                    raise ValueError(f"{name} is {getattr(self, name).shape}, expected {expected}")
            faces = 2 * int(self.n_per_hemi) - 4  # a closed triangulated sphere's face count
            for name in ("tri_in", "tri_out"):
                triangle = getattr(self, name)
                if triangle.size and (triangle.min() < 0 or triangle.max() >= faces):
                    raise ValueError(
                        f"{name}: triangle indices must lie in 0..{faces - 1} within a "
                        f"hemisphere; got {triangle.min()}..{triangle.max()}"
                    )

    @property
    def n_streamlines(self) -> int:
        """How many streamlines these endpoints describe."""
        return int(np.asarray(self.vtx_in).size)

    def take(self, indices) -> Endpoints:
        """The streamlines at ``indices``: a subset, as two halves for split-half reliability.

        Examples
        --------
        >>> both = Endpoints([0, 1, 1], [1, 0, 1], [3, 5, 7], [2, 4, 6], n_per_hemi=10)
        >>> both.take([0, 2]).vtx_in.tolist(), both.take([0, 2]).n_streamlines
        ([3, 7], 2)
        """
        index = np.asarray(indices)
        if index.size == 0:
            index = index.astype(np.intp)  # [] is float64 to NumPy, and takes nothing
        elif index.dtype.kind not in "biu":
            raise TypeError(
                f"streamlines are taken by integer indices or a boolean mask, got {index.dtype}"
            )
        if index.dtype == bool:
            if index.shape != (self.n_streamlines,):
                raise ValueError(f"a mask over {self.n_streamlines} streamlines, got {index.shape}")
        elif (
            index.dtype.kind in "iu"
            and index.shape == (self.n_streamlines,)
            and self.n_streamlines > 2
            and np.isin(index, (0, 1)).all()
        ):
            raise ValueError(
                "0s and 1s, one per streamline, are a mask written as integers, which would be "
                "read as indices: streamlines 0 and 1 over and over. Pass mask.astype(bool), or "
                "np.flatnonzero(mask) for the indices"
            )
        elif index.size and (
            index.min() < -self.n_streamlines or index.max() >= self.n_streamlines
        ):
            raise IndexError(f"streamline indices outside 0..{self.n_streamlines - 1}")

        def pick(values):
            return None if values is None else np.asarray(values)[index]

        return Endpoints(
            surf_in=self.surf_in[index],
            surf_out=self.surf_out[index],
            vtx_in=self.vtx_in[index],
            vtx_out=self.vtx_out[index],
            n_per_hemi=self.n_per_hemi,
            tri_in=pick(self.tri_in),
            tri_out=pick(self.tri_out),
            bary_in=pick(self.bary_in),
            bary_out=pick(self.bary_out),
        )

    @classmethod
    def from_matlab(cls, path, n_per_hemi: int = spec.N_VERTICES_PER_HEMI) -> Endpoints:
        """Read ``mesh_intersections_*.mat`` from the legacy pipeline.

        MATLAB vertex indices are one-based and are converted here.

        This file is the pipeline's *other* branch: the unsnapped filtered
        intersections mapped barycentrically onto the grid. The pipeline's
        smoothed connectome is built from the snapped streamlines on the
        subject's registered high-resolution sphere instead, so endpoints read
        from here do not re-smooth to it: on the same 200,000 streamlines they
        reach r = 0.970 where the coordinates the pipeline fed its smoother
        reach 0.998 (PORTING.md item 6). When ``snapped_fibers.npz`` and the
        registered spheres are at hand, :meth:`from_snapped` gives the
        endpoints that belong with the connectome.
        """
        import scipy.io

        data = scipy.io.loadmat(path)
        missing = [k for k in ("surf_in", "surf_out", "vtx_in", "vtx_out") if k not in data]
        if missing:
            raise FormatError(f"{path} has no {', '.join(missing)}")
        # the raw values go to the constructor, which checks them before it casts
        optional = {}
        if all(k in data for k in ("tri_in", "tri_out", "pt_in", "pt_out")):
            optional = {
                "tri_in": np.asarray(data["tri_in"]).ravel() - 1,
                "tri_out": np.asarray(data["tri_out"]).ravel() - 1,
                "bary_in": np.asarray(data["pt_in"]),
                "bary_out": np.asarray(data["pt_out"]),
            }
        return cls(
            surf_in=np.asarray(data["surf_in"]).ravel(),
            surf_out=np.asarray(data["surf_out"]).ravel(),
            vtx_in=np.asarray(data["vtx_in"]).ravel() - 1,
            vtx_out=np.asarray(data["vtx_out"]).ravel() - 1,
            n_per_hemi=n_per_hemi,
            **optional,
        )

    @classmethod
    def from_snapped(cls, snapped, lh_sphere, rh_sphere, grids=None) -> Endpoints:
        """Endpoints from the pipeline's snapped streamlines and the subject's registered sphere.

        This is the branch the pipeline's smoothed connectome comes from:
        ``snapped_fibers.npz`` gives each streamline's two ends as a surface
        (``0`` left white, ``1`` right white, higher values other surfaces)
        and a vertex of that native surface, and the subject's registered
        sphere, which has one vertex per native vertex, says where each lands
        on the template sphere. Streamlines with an end on any other surface
        are dropped, as ``intersections_to_sphere.py`` drops them, and each
        end is then located on the grid with its triangle and barycentric
        weights.

        Parameters
        ----------
        snapped
            Path to ``snapped_fibers.npz``, or a mapping with ``surf_ids0``,
            ``surf_ids1``, ``v_ids0`` and ``v_ids1``.
        lh_sphere, rh_sphere
            ``(n_native, 3)`` vertex coordinates of the subject's registered
            spheres, any radius, **in the grid's frame**. The pipeline's
            ``?h_sphere_reg_lps.vtk`` are; a sphere stored in RAS, such as
            FreeSurfer's ``sphere.reg``, needs its x and y negated first.
        grids
            ``(lh_grid, rh_grid)`` to locate on; the bundled ico4 grids by
            default.

        The released young adult files were built this way (PORTING.md item
        6).
        """
        from .alignment import normalize_rows
        from .conseal import EndpointConnectome, default_grids

        if isinstance(snapped, (str, bytes)) or hasattr(snapped, "__fspath__"):
            with np.load(snapped) as loaded:
                snapped = {key: loaded[key] for key in loaded.files}
        missing = [k for k in ("surf_ids0", "surf_ids1", "v_ids0", "v_ids1") if k not in snapped]
        if missing:
            raise FormatError(f"the snapped streamlines have no {', '.join(missing)}")
        surf_in = np.asarray(snapped["surf_ids0"]).astype(np.int64).ravel()
        surf_out = np.asarray(snapped["surf_ids1"]).astype(np.int64).ravel()
        vertex_in = np.asarray(snapped["v_ids0"]).astype(np.int64).ravel()
        vertex_out = np.asarray(snapped["v_ids1"]).astype(np.int64).ravel()
        keep = (surf_in >= 0) & (surf_in <= 1) & (surf_out >= 0) & (surf_out <= 1)
        spheres = [normalize_rows(np.asarray(s, dtype=np.float64)) for s in (lh_sphere, rh_sphere)]

        def place(surface, vertex):
            points = np.empty((surface.size, 3))
            for side, sphere in enumerate(spheres):
                pick = surface == side
                # A negative id would wrap round to the far end of the sphere.
                outside = pick & ((vertex < 0) | (vertex >= sphere.shape[0]))
                if outside.any():
                    raise ValueError(
                        f"vertex {int(vertex[outside][0])} is outside the "
                        f"{'left' if side == 0 else 'right'} sphere's {sphere.shape[0]} "
                        f"vertices (0 to {sphere.shape[0] - 1})"
                    )
                points[pick] = sphere[vertex[pick]]
            return points

        lh_grid, rh_grid = grids if grids is not None else default_grids()
        located = EndpointConnectome.from_points(
            lh_grid,
            rh_grid,
            place(surf_in[keep], vertex_in[keep]),
            place(surf_out[keep], vertex_out[keep]),
            surf_in[keep],
            surf_out[keep],
        )
        return located.to_endpoints()

    @property
    def has_positions(self) -> bool:
        """Whether continuous positions are available, not only vertices."""
        return self.bary_in is not None and self.bary_out is not None

    def copy(self) -> Endpoints:
        """An independent copy of every array."""
        from dataclasses import fields as dataclass_fields

        values = {}
        for item in dataclass_fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.copy() if isinstance(value, np.ndarray) else value
        return Endpoints(**values)

    @property
    def global_vertex_in(self) -> np.ndarray:
        """Endpoint vertices as indices into the whole 5124-vertex grid.

        The hemisphere flags are int8, so they are widened before being scaled
        by the vertices per hemisphere -- otherwise the offset overflows.
        """
        return np.asarray(self.vtx_in, dtype=np.int64) + self.global_hemisphere_offset(
            self.surf_in, self.n_per_hemi
        )

    @property
    def global_vertex_out(self) -> np.ndarray:
        """Endpoint vertices as indices into the whole 5124-vertex grid."""
        return np.asarray(self.vtx_out, dtype=np.int64) + self.global_hemisphere_offset(
            self.surf_out, self.n_per_hemi
        )

    @staticmethod
    def global_hemisphere_offset(surf: np.ndarray, per_hemi: int) -> np.ndarray:
        """``surf * per_hemi`` in a width that cannot overflow."""
        return np.asarray(surf, dtype=np.int64) * int(per_hemi)

    @classmethod
    def from_global(
        cls,
        vertex_in: np.ndarray,
        vertex_out: np.ndarray,
        n_per_hemi: int = spec.N_VERTICES_PER_HEMI,
        n_faces_per_hemi: int = spec.N_FACES_PER_HEMI,
        triangle_in: np.ndarray | None = None,
        triangle_out: np.ndarray | None = None,
        barycentric_in: np.ndarray | None = None,
        barycentric_out: np.ndarray | None = None,
    ) -> Endpoints:
        """Build from whole-grid indices, as the HDF5 file stores them.

        The hemisphere is read off the vertex index rather than carried
        separately, and the triangle index has to put its endpoint in the same
        hemisphere, so the stored indices cannot disagree with each other.
        """
        vertex_in = _indices("vertex_in", vertex_in)
        vertex_out = _indices("vertex_out", vertex_out)
        for name, vertex in (("vertex_in", vertex_in), ("vertex_out", vertex_out)):
            if vertex.size and (vertex.min() < 0 or vertex.max() >= 2 * n_per_hemi):
                raise ValueError(
                    f"{name} out of range for a {2 * n_per_hemi}-vertex grid: "
                    f"[{vertex.min()}, {vertex.max()}]"
                )
        optional: dict[str, np.ndarray] = {}
        given = (triangle_in, triangle_out, barycentric_in, barycentric_out)
        if any(item is not None for item in given):
            if any(item is None for item in given):
                raise ValueError(
                    "positions need all of triangle_in, triangle_out, "
                    "barycentric_in and barycentric_out"
                )
            triangle_in = _indices("triangle_in", triangle_in)
            triangle_out = _indices("triangle_out", triangle_out)
            for name, triangle, vertex in (
                ("triangle_in", triangle_in, vertex_in),
                ("triangle_out", triangle_out, vertex_out),
            ):
                if triangle.shape != vertex.shape:
                    raise ValueError(
                        f"{name} has {triangle.size} entries for {vertex.size} streamlines"
                    )
                if triangle.size and (triangle.min() < 0 or triangle.max() >= 2 * n_faces_per_hemi):
                    raise ValueError(
                        f"{name} out of range for a {2 * n_faces_per_hemi}-face grid: "
                        f"[{triangle.min()}, {triangle.max()}]"
                    )
                # Two indices are stored per endpoint; they must agree on the hemisphere.
                if np.any(triangle // n_faces_per_hemi != vertex // n_per_hemi):
                    raise ValueError(
                        f"{name} puts a triangle in the other hemisphere from its vertex"
                    )
            optional = {
                "tri_in": triangle_in % n_faces_per_hemi,
                "tri_out": triangle_out % n_faces_per_hemi,
                "bary_in": _weights("barycentric_in", barycentric_in),
                "bary_out": _weights("barycentric_out", barycentric_out),
            }
        return cls(
            surf_in=(vertex_in // n_per_hemi).astype(np.int8),
            surf_out=(vertex_out // n_per_hemi).astype(np.int8),
            vtx_in=vertex_in % n_per_hemi,
            vtx_out=vertex_out % n_per_hemi,
            n_per_hemi=n_per_hemi,
            **optional,
        )

    def adjacency(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Count matrices ``(A11, A22, A12)`` for LL, RR and LR connections.

        ``A11`` and ``A22`` are symmetrized by adding their transpose, which
        doubles any self-connection on the diagonal; that is what the reference
        does and what the released data was built with.
        """
        n = self.n_per_hemi
        for name, vtx in (("vtx_in", self.vtx_in), ("vtx_out", self.vtx_out)):
            if vtx.size and (vtx.min() < 0 or vtx.max() >= n):
                raise ValueError(
                    f"{name} out of range for {n} vertices per hemisphere: "
                    f"[{vtx.min()}, {vtx.max()}]. One-based indices need converting."
                )

        def counts(rows, cols, shape):
            # bincount over the flattened pair index: exact, and an order of
            # magnitude faster than np.add.at on a million endpoints.
            flat = np.bincount(rows * shape[1] + cols, minlength=shape[0] * shape[1])
            return flat.reshape(shape).astype(np.float64)

        left_left = (self.surf_in == 0) & (self.surf_out == 0)
        a11 = counts(self.vtx_in[left_left], self.vtx_out[left_left], (n, n))
        a11 = a11 + a11.T

        right_right = (self.surf_in == 1) & (self.surf_out == 1)
        a22 = counts(self.vtx_in[right_right], self.vtx_out[right_right], (n, n))
        a22 = a22 + a22.T

        left_right = (self.surf_in == 0) & (self.surf_out == 1)
        a12 = counts(self.vtx_in[left_right], self.vtx_out[left_right], (n, n))
        right_left = (self.surf_in == 1) & (self.surf_out == 0)
        a12 = a12 + counts(self.vtx_in[right_left], self.vtx_out[right_left], (n, n)).T

        return a11, a22, a12


def smooth_endpoints(
    endpoints: Endpoints,
    kernel_left: np.ndarray,
    kernel_right: np.ndarray,
) -> np.ndarray:
    """Smooth endpoint counts into a density, as ``K A K'`` per block.

    Returns the full ``(2 * n_per_hemi, 2 * n_per_hemi)`` symmetric density,
    left hemisphere first. It sums to one over the whole matrix.

    Negative entries are clipped to zero, and the normalizing constant is
    formed from the positive entries only, exactly as in
    ``rdk_smoothed_concon_compute.m``: handed the reference's kernels, this is
    the reference. :func:`smooth` hands it the grid's own kernels, each column
    scaled to unit mass (:func:`kernel_matrices`): nonnegative, so that there
    the clip removes nothing, and no longer symmetric, so that the transpose
    matters.
    """
    a11, a22, a12 = endpoints.adjacency()
    n = endpoints.n_per_hemi

    for name, kernel in (("kernel_left", kernel_left), ("kernel_right", kernel_right)):
        if kernel.shape != (n, n):
            raise ValueError(f"{name} has shape {kernel.shape}, expected {(n, n)}")
        if not np.isfinite(kernel).all():
            raise ValueError(f"{name} is not finite: the kernel's settings overflow it")
    if endpoints.n_streamlines == 0:
        raise ValueError("there are no streamlines to smooth: the endpoints are empty")

    # K A K', which the reference writes K A K: its kernels are symmetric, and a kernel
    # whose columns are scaled to unit mass is not
    im11 = kernel_left @ a11 @ kernel_left.T
    im22 = kernel_right @ a22 @ kernel_right.T
    im12 = kernel_left @ a12 @ kernel_right.T

    total = im11[im11 > 0].sum() + im22[im22 > 0].sum() + 2.0 * im12[im12 > 0].sum()
    if not np.isfinite(total):
        raise ValueError(
            "the smoothed intensity is not finite: the kernel's values are too large to "
            "multiply; the kernel's settings, not the endpoints, are the cause"
        )
    if total <= 0:
        raise ValueError(
            "the smoothed intensity has no positive mass: the kernel is zero everywhere, or "
            "negative, where the endpoints are; the kernel's settings, not the endpoints, are "
            "the cause"
        )

    density = np.zeros((2 * n, 2 * n), dtype=np.float64)
    density[:n, :n] = np.where(im11 > 0, im11 / total, 0.0)
    density[n:, n:] = np.where(im22 > 0, im22 / total, 0.0)
    block12 = np.where(im12 > 0, im12 / total, 0.0)
    density[:n, n:] = block12
    density[n:, :n] = block12.T
    return density


def load_eigenpairs(path, hemisphere: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``EV_LBO_ds_ico4_{L,R}.mat`` and return ``(eigenvalues, eigenvectors)``.

    A file that cannot be read, or holds other variables, is refused with its
    path and the way to fetch the SBCI_Toolkit's copy again, ``sbci basis
    --force``: an empty or cut file raised SciPy's own error, naming neither.
    """
    import scipy.io

    hemisphere = hemisphere.upper()
    if hemisphere not in ("L", "R"):
        raise ValueError(f"hemisphere must be 'L' or 'R', got {hemisphere!r}")

    again = "if it is the SBCI_Toolkit's basis, fetch it again with `sbci basis --force`"
    try:
        data = scipy.io.loadmat(path)
    except Exception as error:  # an empty, cut or foreign file fails in many ways
        raise FormatError(
            f"{path} cannot be read as a MATLAB file ({type(error).__name__}: {error}): it is "
            f"damaged or not the Laplace-Beltrami basis; {again}"
        ) from None
    lam_key, vec_key = f"Lambda_{hemisphere}", f"U_{hemisphere}"
    if lam_key not in data or vec_key not in data:
        available = [k for k in data if not k.startswith("__")]
        raise FormatError(f"{path} has no {lam_key}/{vec_key}; it has {available}; {again}")

    eigenvalues = np.asarray(data[lam_key], dtype=np.float64).ravel()
    eigenvectors = np.asarray(data[vec_key], dtype=np.float64)
    if not np.all(np.diff(eigenvalues) >= 0):
        raise ValueError(f"{path}: eigenvalues are not ascending")
    return eigenvalues, eigenvectors


_NO_ENDPOINTS = (
    "This file carries no streamline endpoints, so there is nothing to "
    "re-smooth from. The format has an optional /endpoints group: build one "
    "with Endpoints.from_snapped('snapped_fibers.npz', lh_sphere, rh_sphere), "
    "from the pipeline's snapped streamlines on the subject's registered "
    "spheres, which re-smooth to the pipeline's connectome (the unsnapped "
    "intersections Endpoints.from_matlab reads do not), and pass it to "
    "ContinuousConnectome.save(), or construct the connectome with "
    "endpoints=... before saving."
)

_NOT_STRUCTURAL = (
    "Only a structural connectome can be re-smoothed, from the streamline endpoints "
    "it carries; this one is {modality!r}, and a functional connectome has no "
    "streamlines to smooth."
)

_NO_POSITIONS = (
    "The spherical kernel smooths from where each streamline actually crossed "
    "the surface, so it needs the barycentric coordinates in the /endpoints "
    "group, not only the nearest vertex. This file carries vertices alone. "
    "Rebuild it with Endpoints.from_snapped('snapped_fibers.npz', lh_sphere, "
    "rh_sphere), which keeps them, or pass kernel='rdk'."
)

EIGENPAIR_FILES = ("EV_LBO_ds_ico4_L.mat", "EV_LBO_ds_ico4_R.mat")
"""The SBCI_Toolkit's basis files: what :func:`find_basis` looks for, and what
``eigenpairs=`` reads when it names a directory."""

EIGENPAIR_ENV = "SBCI_LBO_DIR"
"""Environment variable naming the directory that holds the SBCI_Toolkit's basis."""

SIGNED_LIMIT = 1e-3
"""The share of a kernel's weight that may be negative before :func:`smooth` warns.

The grid's own basis gives 1e-13, rounding; the SBCI_Toolkit's gives 0.37 at
the reference's bandwidth and 0.04 at 18.9 mm (A2-02)."""

SHK_WIDTH_FLOOR = 0.001
"""The ``sigma`` below which :func:`smooth` warns that the 33 terms, not
``sigma``, set the spherical kernel's width (:func:`spherical_heat_kernel`)."""


def _candidate_basis_directories():
    """Where :func:`find_basis` looks, in order."""
    import os
    from pathlib import Path

    candidates = []
    if os.environ.get(EIGENPAIR_ENV):
        candidates.append(Path(os.environ[EIGENPAIR_ENV]))
    if os.environ.get("SBCI_TOOLKIT"):
        candidates.append(Path(os.environ["SBCI_TOOLKIT"]) / "concon_estimate")
    candidates.append(Path.home() / ".cache" / "sbci" / "lbo")
    candidates.append(Path.home() / ".sbci" / "lbo")
    return candidates


@lru_cache(maxsize=8)
def _file_digest(path: str, size: int, modified: int) -> str:
    """SHA-256 of a file, kept while its size and time stamp stay the same."""
    from pathlib import Path

    from .download import sha256_of

    return sha256_of(Path(path))


def _digest(path) -> str:
    """SHA-256 of a basis file, read once while it is unchanged."""
    status = path.stat()
    return _file_digest(str(path), status.st_size, status.st_mtime_ns)


def _basis_problem(path) -> str | None:
    """Why a basis file is not the SBCI_Toolkit's, or ``None`` if its size and SHA-256 are."""
    from .download import BASIS_FILES

    size, digest = BASIS_FILES[path.name]
    found = path.stat().st_size
    if found != size:
        return f"{path} is {found:,} bytes, not the {size:,} of the SBCI_Toolkit's"
    actual = _digest(path)
    if actual != digest:
        return f"{path} has SHA-256 {actual[:16]}..., not the SBCI_Toolkit's {digest[:16]}..."
    return None


def find_basis():
    """The folder holding the SBCI_Toolkit's Laplace-Beltrami basis, checked, or ``None``.

    :func:`smooth` does not need it: ``rdk`` and ``matern`` build the grid's
    own basis (:func:`laplace_beltrami_basis`). The toolkit's files,
    ``EV_LBO_ds_ico4_L.mat`` and ``EV_LBO_ds_ico4_R.mat``, 50 MB a hemisphere,
    are what the MATLAB smoothed with, and serve comparisons with it;
    ``sbci basis`` fetches them. This looks in ``$SBCI_LBO_DIR``,
    ``$SBCI_TOOLKIT/concon_estimate``, ``~/.cache/sbci/lbo`` and
    ``~/.sbci/lbo``, in that order, and returns the first folder holding both
    files at the size and SHA-256 that :data:`sbci.download.BASIS_FILES` pins.

    Raises
    ------
    MissingDataError
        When the only files found are damaged or not the toolkit's. It names
        each, and the fix: ``sbci basis --force``.
    """
    damaged = []
    for directory in _candidate_basis_directories():
        paths = [directory / name for name in EIGENPAIR_FILES]
        if not all(path.is_file() for path in paths):
            continue
        problems = [problem for path in paths if (problem := _basis_problem(path))]
        if not problems:
            return directory
        damaged.extend(problems)
    if damaged:
        raise MissingDataError(
            "The Laplace-Beltrami basis files found are damaged, or not the SBCI_Toolkit's: "
            + "; ".join(damaged)
            + ". Fetch them again with `sbci basis --force`."
        )
    return None


def _resolve_eigenpairs(eigenpairs):
    """Accept a directory of MATLAB files or a pair of arrays; return both."""
    import os
    from pathlib import Path

    if isinstance(eigenpairs, (str, os.PathLike)):
        base = Path(eigenpairs)
        if not base.is_dir():
            raise NotADirectoryError(
                f"{base} is not a directory; eigenpairs= should name the folder "
                f"holding {EIGENPAIR_FILES[0]} and {EIGENPAIR_FILES[1]}"
            )
        paths = [base / name for name in EIGENPAIR_FILES]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing eigenpair files: {missing}")
        return load_eigenpairs(paths[0], "L"), load_eigenpairs(paths[1], "R")

    try:
        (lam_left, vec_left), (lam_right, vec_right) = eigenpairs
    except (TypeError, ValueError) as error:
        raise TypeError(
            "eigenpairs= should be None, a directory, or a pair of "
            "(eigenvalues, eigenvectors) tuples for left and right"
        ) from error
    return (
        (np.asarray(lam_left, dtype=np.float64).ravel(), np.asarray(vec_left, dtype=np.float64)),
        (np.asarray(lam_right, dtype=np.float64).ravel(), np.asarray(vec_right, dtype=np.float64)),
    )


def _basis(eigenpairs):
    """``(left, right, which)``: the eigenpairs ``rdk`` and ``matern`` use, and a line naming them.

    The line goes into the result's provenance: the grid's own basis, or the
    SHA-256 of each file read, or of the arrays passed.
    """
    import hashlib
    import os
    from pathlib import Path

    if eigenpairs is None:
        left, right = (laplace_beltrami_basis(side)[:2] for side in spec.HEMISPHERE_ORDER)
        return left, right, "basis the grid's own Laplace-Beltrami eigenpairs"
    left, right = _resolve_eigenpairs(eigenpairs)
    if isinstance(eigenpairs, (str, os.PathLike)):
        files = ", ".join(
            f"{name} sha256 {_digest(Path(eigenpairs) / name)}" for name in EIGENPAIR_FILES
        )
        return left, right, f"basis {files}"
    digest = hashlib.sha256()
    for array in (*left, *right):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return left, right, f"basis eigenpairs passed as arrays, sha256 {digest.hexdigest()}"


def _settings(kernel: str, bandwidth: float, nu: float | None = None) -> str:
    """The kernel's settings, as an error message names them."""
    return f"kernel={kernel!r}, bandwidth={bandwidth:g}" + ("" if nu is None else f", nu={nu:g}")


def kernel_matrices(kernel="rdk", bandwidth=None, area=None, eigenpairs=None, nu=None):
    """The two kernel matrices :func:`smooth` applies for ``rdk`` and ``matern``: ``(left, right)``.

    Column ``j`` of each is the bump an endpoint at vertex ``j`` of that
    hemisphere spreads over it, scaled to integrate to one under ``area``:
    ``K / (area @ K)``, so ``area @ K`` is 1 in every column, and every
    streamline end carries the same mass wherever it lands.

    Parameters
    ----------
    kernel
        ``"rdk"`` or ``"matern"``.
    bandwidth
        ``kappa``, in mm on the white surface; :data:`DEFAULT_KAPPA` if
        ``None``.
    area
        Per-vertex weights over the whole grid, left hemisphere first: a
        file's ``area``. By default the format's own, counts of fsaverage
        vertices per grid vertex (57 to 68).
    eigenpairs
        ``None``, the default, for the grid's own basis
        (:func:`laplace_beltrami_basis`); or a directory of the SBCI_Toolkit's
        files, or a pair of ``(eigenvalues, eigenvectors)`` tuples, as
        :func:`smooth` takes them.
    nu
        The Matern smoothness, :data:`DEFAULT_NU` if ``None``; ``matern`` only.

    Examples
    --------
    >>> import numpy as np
    >>> left, right = kernel_matrices("rdk")                      # doctest: +SKIP
    >>> from sbci.io.cifti import vertex_areas
    >>> bool(np.allclose(vertex_areas()[1][:2562] @ left, 1.0))  # doctest: +SKIP
    True
    """
    if kernel not in ("rdk", "matern"):
        raise ValueError(f"kernel must be 'rdk' or 'matern', got {kernel!r}")
    bandwidth = DEFAULT_KAPPA[kernel] if bandwidth is None else _positive("bandwidth", bandwidth)
    if nu is not None and kernel != "matern":
        raise ValueError(f"nu is the Matern kernel's smoothness, and kernel={kernel!r} has none")
    if kernel == "matern":
        nu = DEFAULT_NU if nu is None else _positive("nu", nu)
    left, right, _ = _basis(eigenpairs)
    return _kernel_pair(kernel, bandwidth, area, left, right, nu)


def _kernel_pair(kernel, bandwidth, area, left, right, nu=None):
    """Both hemispheres' kernels, each column scaled to unit mass under ``area``; checked."""
    if area is None:
        from .io.cifti import vertex_areas

        area = vertex_areas()[1]
    area = np.asarray(area, dtype=np.float64).ravel()
    sizes = [np.shape(vectors)[0] for _, vectors in (left, right)]
    if area.size != sum(sizes):
        raise ValueError(
            f"{area.size} vertex weights for a basis of {sizes[0]} + {sizes[1]} vertices"
        )
    settings = _settings(kernel, bandwidth, nu)
    build = diffusion_kernel if kernel == "rdk" else partial(matern_kernel, nu=nu)
    pair, negative, total = [], 0.0, 0.0
    for side, (eigenvalues, eigenvectors), weights in (
        ("left", left, area[: sizes[0]]),
        ("right", right, area[sizes[0] :]),
    ):
        matrix = build(eigenvalues, eigenvectors, bandwidth)
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"at {settings} the {side} hemisphere's kernel is not finite; the settings, not "
                "the endpoints, are the cause"
            )
        mass = weights @ matrix
        empty = ~(mass > 0)
        if empty.any():
            raise ValueError(
                f"at {settings} the kernel is zero everywhere, or negative overall, for "
                f"{int(empty.sum())} of the {side} hemisphere's {mass.size} vertices: an endpoint "
                "there would carry no mass. The settings or the basis, not the endpoints, are the "
                "cause"
            )
        matrix /= mass[np.newaxis, :]
        negative -= float((weights @ np.minimum(matrix, 0.0)).sum())
        total += float((weights @ np.abs(matrix)).sum())
        pair.append(matrix)
    if total > 0 and negative / total > SIGNED_LIMIT:
        warnings.warn(
            f"at {settings}, {negative / total:.1%} of the kernel's weight is negative, so it is "
            "not a smoothing kernel, and the density is clipped at zero. A basis orthonormal "
            "under the surface's vertex areas, as a Laplace-Beltrami basis is, makes none; the "
            "SBCI_Toolkit's EV_LBO_ds_ico4 files are not one, and eigenpairs=None builds the "
            "grid's own",
            UserWarning,
            stacklevel=3,
        )
    return pair[0], pair[1]


def _positive(name: str, value) -> float:
    """A number :func:`smooth` takes, as a float, refused unless it is positive and finite.

    ``True`` and ``"0.005"`` are refused before the cast, which reads them as
    1.0 and 0.005. A NumPy float counts as the decimal it prints as: a float32
    0.005 is 0.004999999888 as a float, which a file recorded and the cohort
    loader told apart from the 0.005 of every other file.
    """
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise TypeError(f"{name} takes a number, got {value!r}")
    number = float(str(value)) if isinstance(value, np.floating) else float(value)
    if not (number > 0 and np.isfinite(number)):
        raise ValueError(f"{name} must be positive and finite, got {number}")
    return number


def smooth(
    connectome,
    kernel: str = "shk",
    bandwidth: float | None = None,
    eigenpairs=None,
    mask_medial_wall: bool = False,
    progress=None,
    nu: float | None = None,
):
    """Re-smooth a structural connectome from its stored streamline endpoints.

    Returns a new connectome of the same type, carrying the same endpoints,
    mask, areas and coordinates, with ``kernel``, ``bandwidth`` (and ``nu``
    for ``matern``) and the streamline count in its metadata set to what was
    actually used -- a half of a subject's streamlines records its own count --
    and its provenance saying how it was re-smoothed.

    The density is normalized to the package's convention, ``area @ D @ area
    == 1`` with the diagonal dropped, so that the result satisfies the same
    unit-mass check as a file read from disk. ``area``, the format's weight per
    vertex, is a count of fsaverage vertices, 57 to 68 a grid vertex
    (SPEC_QUESTIONS.md item 3), not a surface area. ``rdk`` and ``matern``
    scale each endpoint's kernel to unit mass under these counts. ``shk``
    integrates its kernel as ``concon`` did, so under them the mass a
    streamline end receives depends on where it lands: averaged over each
    Desikan region it runs from 0.918 to 1.035, where the sphere's own vertex
    areas would give 0.996 to 1.003.

    Parameters
    ----------
    connectome
        A structural :class:`~sbci.ContinuousConnectome` carrying endpoints.
    kernel
        One of :data:`KERNELS`.
    bandwidth
        For ``shk``, ``sigma``, a diffusion time on the unit sphere, by
        default :data:`DEFAULT_SIGMA`, 0.005: the kernel falls to half height
        6.0 degrees from an endpoint and to zero at 12.2 degrees, about 7 and
        15 mm on the white surface. Below 0.001 the 33-term series, not
        ``sigma``, sets the width, and a warning says so. For ``rdk`` and
        ``matern``, ``kappa``, in mm on the white surface, by default
        :data:`DEFAULT_KAPPA`, where each smooths at ``shk``'s default scale.
        A positive finite number; a NumPy float counts as the decimal it
        prints as, so ``np.float32(0.005)`` is smoothed and recorded as 0.005.
    eigenpairs
        The basis ``rdk`` and ``matern`` use. ``None``, the default, builds
        the grid's own (:func:`laplace_beltrami_basis`), under half a minute
        once a session, with nothing to download. A directory
        holding the SBCI_Toolkit's ``EV_LBO_ds_ico4_L.mat`` and
        ``EV_LBO_ds_ico4_R.mat``, or a pair of ``(eigenvalues,
        eigenvectors)`` tuples for left and right, is used as given and named
        in the provenance by its SHA-256; one that makes a signed kernel, as
        the toolkit's does, draws a warning.
    mask_medial_wall
        ``False``, the default, leaves what the kernel put on the medial wall,
        as both references do; such a connectome saves, but fails
        ``sbci validate``'s mask check. ``True`` zeroes the wall's rows and
        columns before the unit-mass normalization, so that the result
        satisfies the format's mask rule and ``sbci validate``, as every
        released file does -- SPEC_QUESTIONS.md item 14. It does not drop the
        streamlines that end on the wall: their kernels still reach the cortex
        nearby, and with ``shk`` an end on the wall keeps 12% of its mass there
        on average.
    progress
        Optional ``progress(done, total)`` callable, called after each block of
        endpoints for the ``shk`` kernel.
    nu
        The Matern kernel's smoothness, a positive number, recorded in the
        metadata as ``nu``; ``None`` means :data:`DEFAULT_NU`, the reference
        script's 3 (it names 1, 2 and 3). Only ``matern`` takes it.

    Raises
    ------
    ValueError
        For a functional connectome, which has no streamlines; and for kernel
        settings that make no usable kernel, naming them.
    MissingDataError
        For a connectome without endpoints.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    if bandwidth is not None:
        bandwidth = _positive("bandwidth", bandwidth)
    if nu is not None:
        if kernel != "matern":
            raise ValueError(
                f"nu is the Matern kernel's smoothness, and kernel={kernel!r} has none; "
                f"got nu={nu!r}"
            )
        nu = _positive("nu", nu)
    metadata = getattr(connectome, "metadata", None)
    modality = getattr(metadata, "modality", "sc")
    if modality != "sc":
        raise ValueError(_NOT_STRUCTURAL.format(modality=modality))
    if getattr(connectome, "endpoints", None) is None:
        raise MissingDataError(_NO_ENDPOINTS)

    if kernel == "shk":
        sigma = DEFAULT_SIGMA if bandwidth is None else bandwidth
        if sigma < SHK_WIDTH_FLOOR:
            warnings.warn(
                f"sigma {sigma:g} is below {SHK_WIDTH_FLOOR:g}, where the spherical kernel's 33 "
                "terms, not sigma, set its width: its half height stays 3.6 to 4.0 degrees from "
                "the endpoint, so a smaller sigma no longer makes it narrower",
                UserWarning,
                stacklevel=3,
            )
        vertices = _grid_vertices(connectome)  # off ico4 refused first: positions are on its sphere
        points_in, points_out = endpoint_positions(connectome.endpoints)
        from .grid import hemisphere_labels

        density = endpoint_density(
            vertices,
            points_in,
            points_out,
            connectome.endpoints.surf_in,
            connectome.endpoints.surf_out,
            hemisphere_labels(connectome.n_vertices),
            sigma=sigma,
            progress=progress,
        )
        apply_final_threshold(density, points_in.shape[0])
        return _finish(connectome, density, kernel, sigma, mask_medial_wall)

    left, right, which = _basis(eigenpairs)
    if bandwidth is None:
        bandwidth = DEFAULT_KAPPA[kernel]
    if kernel == "matern" and nu is None:
        nu = DEFAULT_NU
    kernel_left, kernel_right = _kernel_pair(kernel, bandwidth, connectome.area, left, right, nu)
    density = smooth_endpoints(connectome.endpoints, kernel_left, kernel_right)
    return _finish(connectome, density, kernel, bandwidth, mask_medial_wall, nu, which)


def apply_final_threshold(
    density: np.ndarray, n_streamlines: int, threshold: float = FINAL_THRESHOLD
) -> np.ndarray:
    """Drop pairs ``c3_main`` would not have written, in place.

    Its output is the accumulated kernel product divided by the streamline
    count, written only where that exceeds ``--final_thold``. The density here
    is the undivided sum, so the rule is applied on the divided scale.

    Examples
    --------
    >>> d = np.array([[0.0, 3e-9], [3e-9, 0.0]]) * 1000
    >>> apply_final_threshold(d, 1000)            # 3e-9 per streamline survives
    array([[0.e+00, 3.e-06],
           [3.e-06, 0.e+00]])
    >>> apply_final_threshold(d, 10**7)           # 3e-13 per streamline does not
    array([[0., 0.],
           [0., 0.]])
    """
    if n_streamlines <= 0:
        raise ValueError("the threshold is per streamline; there are none")
    density[density / n_streamlines <= threshold] = 0.0
    return density


def _grid_vertices(connectome) -> np.ndarray:
    """The unit sphere the endpoints were located on: the bundled ico4 grid.

    The file's ``/coordinates`` are not used. The format fixes no frame for
    them -- they may hold any geometry -- whereas the endpoints' barycentric
    positions refer to the pipeline's ico4 sphere, which the bundled mesh
    reproduces to 6e-12 (PORTING.md item 6).
    """
    from .surface import load_surface

    if connectome.n_vertices != spec.N_VERTICES:
        raise ValueError(
            f"shk smoothing needs the {spec.N_VERTICES}-vertex ico4 grid, "
            f"not {connectome.n_vertices} vertices"
        )
    vertices = np.asarray(load_surface("sphere").vertices, dtype=np.float64)
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True)


def _finish(
    connectome,
    density: np.ndarray,
    kernel: str,
    bandwidth: float,
    mask_medial_wall: bool = False,
    nu: float | None = None,
    basis: str | None = None,
):
    """Normalize a re-smoothed density and wrap it back into a connectome.

    ``nu`` is recorded for the Matern kernel, :data:`DEFAULT_NU` if not given;
    any other kernel leaves the key out, whatever the connectome carried. The
    provenance is rewritten to say how the density was made (:func:`_provenance`),
    ``basis`` naming the basis ``rdk`` and ``matern`` used. A density that is
    not finite, or carries no positive mass, is refused with the kernel's
    settings, before anything could save it.
    """
    from .grid import to_condensed
    from .metadata import Metadata

    if kernel == "matern" and nu is None:
        nu = DEFAULT_NU
    settings = _settings(kernel, bandwidth, nu if kernel == "matern" else None)
    if not np.isfinite(density).all():
        raise ValueError(
            f"at {settings} the smoothed density is not finite; the settings, not the endpoints, "
            "are the cause"
        )
    # The medial wall is NOT zeroed by default. Neither reference masks -- c3_main's
    # own output and the MATLAB rdk density both leave mass there -- and the default
    # follows them, though a re-smoothed density can then fail the validator's mask
    # check: SPEC_QUESTIONS.md item 14. `mask_medial_wall=True` takes the format's side.
    if mask_medial_wall:
        wall = ~np.asarray(connectome.mask, dtype=bool)
        density[wall, :] = 0.0
        density[:, wall] = 0.0

    # The storage convention is the strict upper triangle, so self-connectivity
    # is dropped (SPEC_QUESTIONS.md item 2). Normalize after dropping it, or
    # the file that comes back would not satisfy the unit-mass check that
    # validates it.
    np.fill_diagonal(density, 0.0)
    area = np.asarray(connectome.area, dtype=np.float64)
    mass = float(area @ density @ area)
    if not (mass > 0 and np.isfinite(mass)):
        where = " once the medial wall is zeroed" if mask_medial_wall else ""
        raise ValueError(
            f"at {settings} the smoothed density has no positive mass under the file's vertex "
            f"weights (counts of fsaverage vertices){where}"
        )
    density /= mass

    fields = copy.deepcopy(connectome.metadata.fields)  # not sharing its lists with the input
    fields.update(kernel=kernel, bandwidth=float(bandwidth), normalization="unit-mass")
    fields.pop("nu", None)
    if kernel == "matern":
        fields["nu"] = float(nu)
    if getattr(connectome, "endpoints", None) is not None:
        fields["streamline_count"] = int(connectome.endpoints.n_streamlines)
    fields["provenance"] = _provenance(
        fields.get("provenance"), kernel, bandwidth, nu if kernel == "matern" else None, basis
    )
    return type(connectome)(
        data=to_condensed(density).astype(np.float32),
        area=connectome.area,
        mask=connectome.mask,
        metadata=Metadata(fields),
        coords=connectome.coords,
        endpoints=connectome.endpoints,
    )


_SMOOTHED = re.compile(r"(?:re-)?smoothed by sbci\b[^;]*")
"""A provenance's clause on how its density was smoothed, up to the next semicolon."""


def _provenance(previous, kernel, bandwidth, nu=None, basis=None) -> str:
    """The provenance of a smoothed connectome: what it said, with the smoothing restated.

    A released file's provenance says how it was smoothed; that clause is
    replaced, rather than left beside another kernel's (A2-09), and one is
    added where there was none. A connectome with no provenance yet, such as
    one :func:`sbci.example` builds, is smoothed for the first time.

    Examples
    --------
    >>> old = "HCP Young Adult sub-x; smoothed by sbci (kernel shk, bandwidth 0.005); built"
    >>> print(_provenance(old, "rdk", 5.5))
    HCP Young Adult sub-x; re-smoothed by sbci ... from its stored endpoints
    (kernel rdk, bandwidth 5.5); built
    """
    from . import __version__

    details = f"kernel {kernel}, bandwidth {bandwidth:g}"
    if nu is not None:
        details += f", nu {nu:g}"
    if basis:
        details += f", {basis}"
    clause = f"re-smoothed by sbci {__version__} from its stored endpoints ({details})"
    text = "" if previous is None else str(previous).strip()
    if not text:
        return f"smoothed by sbci {__version__} from its endpoints ({details})"
    if not _SMOOTHED.search(text):
        return f"{text}; {clause}"
    parts = [part.strip() for part in text.split(";")]
    kept, placed = [], False
    for part in parts:
        if _SMOOTHED.search(part):
            if placed:
                continue
            part, placed = _SMOOTHED.sub(lambda _match: clause, part, count=1), True
        kept.append(part)
    return "; ".join(part for part in kept if part)

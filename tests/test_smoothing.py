"""The kernel smoothing port.

The kernels are checked against closed forms on an orthonormal basis, where
``U diag(rho) U'`` reduces to something writable by hand, and against the
limiting behaviour the maths dictates: as the bandwidth grows, every mode but
the constant one is suppressed and the kernel must collapse to rank one.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from sbci.metadata import template as metadata_template
from sbci.smoothing import (
    TRUNCATION_FLOOR,
    Endpoints,
    diffusion_kernel,
    kappa_candidates,
    matern_kernel,
    smooth_endpoints,
)


@pytest.fixture
def basis():
    """An orthonormal basis and a spectrum, so the kernels have a closed form."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    eigenvalues = np.linspace(0.0, 1.0, 12)
    return eigenvalues, q


# --- the kernels ----------------------------------------------------------


def test_diffusion_kernel_matches_its_closed_form(basis):
    """With an orthonormal U, K = U diag(rho) U' and rho is exp(-k^2 L / 2)."""
    eigenvalues, u = basis
    kappa = 1.7
    expected = u @ np.diag(np.exp(-(kappa**2) / 2 * eigenvalues)) @ u.T
    np.testing.assert_allclose(diffusion_kernel(eigenvalues, u, kappa), expected)


def test_matern_kernel_matches_its_closed_form(basis):
    """The reference's weights, divided by the largest: the scale cancels in the smoothing."""
    eigenvalues, u = basis
    kappa, nu = 1.3, 2.0
    rho = (2 * nu / kappa**2 + eigenvalues) ** (-nu - 1)
    np.testing.assert_allclose(
        matern_kernel(eigenvalues, u, kappa, nu), u @ np.diag(rho / rho.max()) @ u.T
    )


@pytest.mark.parametrize("build", [diffusion_kernel, matern_kernel])
def test_kernels_are_symmetric(basis, build):
    eigenvalues, u = basis
    kernel = build(eigenvalues, u, 1.1)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)


@pytest.mark.parametrize("build", [diffusion_kernel, matern_kernel])
def test_kernels_are_positive_semidefinite(basis, build):
    """A positive rho means U diag(rho) U' has no negative eigenvalue."""
    eigenvalues, u = basis
    assert np.linalg.eigvalsh(build(eigenvalues, u, 1.1)).min() > -1e-12


def test_diffusion_kernel_collapses_to_rank_one(basis):
    """Large bandwidth suppresses every mode but the one at lambda = 0."""
    eigenvalues, u = basis
    kernel = diffusion_kernel(eigenvalues, u, kappa=400.0)
    spectrum = np.linalg.eigvalsh(kernel)
    significant = spectrum > 1e-6 * spectrum.max()
    assert significant.sum() == 1, f"rank {significant.sum()}, expected 1"
    np.testing.assert_allclose(kernel, np.outer(u[:, 0], u[:, 0]), atol=1e-9)


def test_smaller_bandwidth_keeps_more_modes(basis):
    """Monotone in the only sense the maths guarantees: modes retained."""
    eigenvalues, u = basis
    counts = []
    for kappa in (1.0, 10.0, 100.0, 400.0):
        rho = np.exp(-(kappa**2) / 2 * eigenvalues)
        counts.append(int((rho > TRUNCATION_FLOOR).sum()))
    assert counts == sorted(counts, reverse=True), counts


@pytest.mark.parametrize("build", [diffusion_kernel, matern_kernel])
def test_kernels_reject_a_nonpositive_bandwidth(basis, build):
    eigenvalues, u = basis
    with pytest.raises(ValueError, match="kappa must be positive"):
        build(eigenvalues, u, 0.0)


def test_matern_rejects_a_nonpositive_nu(basis):
    eigenvalues, u = basis
    with pytest.raises(ValueError, match="nu must be positive"):
        matern_kernel(eigenvalues, u, 1.0, nu=0.0)


def test_kernel_rejects_a_mismatched_spectrum(basis):
    _, u = basis
    with pytest.raises(ValueError, match="eigenvalues for"):
        diffusion_kernel(np.zeros(5), u, 1.0)


@pytest.mark.parametrize("build", [diffusion_kernel, matern_kernel])
def test_kernels_reject_an_infinite_bandwidth(basis, build):
    """exp(-inf) is a kernel of zeros, which only fails later as 'no positive mass'."""
    eigenvalues, u = basis
    with pytest.raises(ValueError, match="positive and finite"):
        build(eigenvalues, u, np.inf)


def test_descending_eigenvalues_are_refused_where_they_are_consumed(basis):
    """``load_eigenpairs`` checks the files; arrays passed in directly were not checked.

    The default bandwidth reads the spectrum by position, so a descending
    array gave a default kappa of 26,899 where the ascending one gives 1.41.
    """
    eigenvalues, u = basis
    descending, flipped = eigenvalues[::-1], u[:, ::-1]
    with pytest.raises(ValueError, match="ascending"):
        diffusion_kernel(descending, flipped, 1.0)
    with pytest.raises(ValueError, match="ascending"):
        matern_kernel(descending, flipped, 1.0)
    with pytest.raises(ValueError, match="ascending"):
        kappa_candidates(np.linspace(0.54, 0.0, 2562))
    # ties are ascending enough: a repeated eigenvalue is not a reversal
    np.testing.assert_allclose(
        diffusion_kernel(np.array([0.0, 0.5, 0.5]), np.eye(3), 1.0),
        np.diag(np.exp(-0.5 * np.array([0.0, 0.5, 0.5]))),
    )


def test_smooth_refuses_a_descending_basis(smoothable, toy_basis):
    (eigenvalues, u), _ = toy_basis
    reversed_basis = ((eigenvalues[::-1], u[:, ::-1]),) * 2
    with pytest.raises(ValueError, match="ascending"):
        smoothable.smooth(kernel="rdk", eigenpairs=reversed_basis)
    with pytest.raises(ValueError, match="ascending"):
        smoothable.smooth(kernel="rdk", bandwidth=1.0, eigenpairs=reversed_basis)


@pytest.mark.parametrize("bad", [0.0, -0.005, np.nan, np.inf])
def test_smooth_refuses_a_bandwidth_that_is_not_a_positive_number(smoothable, toy_basis, bad):
    """Zero and negative sigma went through and were recorded in the metadata as used."""
    with pytest.raises(ValueError, match="bandwidth must be positive and finite"):
        smoothable.smooth(kernel="shk", bandwidth=bad)
    with pytest.raises(ValueError, match="bandwidth must be positive and finite"):
        smoothable.smooth(kernel="rdk", bandwidth=bad, eigenpairs=toy_basis)


# --- bandwidth selection ---------------------------------------------------


def test_kappa_candidates_are_increasing_and_the_right_count():
    eigenvalues = np.linspace(0.0, 0.54, 2562)
    candidates = kappa_candidates(eigenvalues, n=10)
    assert candidates.size == 10
    assert np.all(np.diff(candidates) > 0)


def test_kappa_candidates_match_the_reference_formula():
    """The bounds are the ones rdk_smoothed_concon_compute.m computes."""
    eigenvalues = np.linspace(0.0, 0.54, 2562)
    candidates = kappa_candidates(eigenvalues, n=10, index=200)
    assert candidates[0] == pytest.approx(np.sqrt(-2 * np.log(0.9) / eigenvalues[-1]))
    assert candidates[-1] == pytest.approx(np.sqrt(-2 * np.log(0.001) / eigenvalues[199]))


def test_kappa_candidates_need_enough_spectrum():
    with pytest.raises(ValueError, match="at least 200 eigenvalues"):
        kappa_candidates(np.linspace(0.0, 1.0, 50))


# --- endpoints -------------------------------------------------------------


@pytest.fixture
def endpoints():
    """Four streamlines over a two-vertex-per-hemisphere toy surface.

    left-left   : vertex 0 <-> vertex 1
    right-right : vertex 0 <-> vertex 0 (a self connection)
    left-right  : left 1 -> right 0
    right-left  : right 1 -> left 0
    """
    return Endpoints(
        surf_in=np.array([0, 1, 0, 1], dtype=np.int8),
        surf_out=np.array([0, 1, 1, 0], dtype=np.int8),
        vtx_in=np.array([0, 0, 1, 1]),
        vtx_out=np.array([1, 0, 0, 0]),
        n_per_hemi=2,
    )


def test_adjacency_is_built_by_hand(endpoints):
    a11, a22, a12 = endpoints.adjacency()

    # One left-left streamline 0->1, symmetrized to both off-diagonal entries.
    np.testing.assert_array_equal(a11, [[0, 1], [1, 0]])

    # One right-right self connection, doubled by the symmetrization -- which
    # is what the reference does, so the released data carries it too.
    np.testing.assert_array_equal(a22, [[2, 0], [0, 0]])

    # left 1 -> right 0, plus right 1 -> left 0 transposed into the same block.
    np.testing.assert_array_equal(a12, [[0, 1], [1, 0]])


def test_adjacency_totals_count_each_within_hemisphere_streamline_twice(endpoints):
    a11, a22, _ = endpoints.adjacency()
    left_left = int(((endpoints.surf_in == 0) & (endpoints.surf_out == 0)).sum())
    assert a11.sum() == 2 * left_left


def test_one_based_indices_are_caught():
    """MATLAB indices are one-based; unconverted, they are refused as Endpoints are made."""
    with pytest.raises(ValueError, match="One-based indices"):
        Endpoints(
            surf_in=np.array([0], dtype=np.int8),
            surf_out=np.array([0], dtype=np.int8),
            vtx_in=np.array([2]),
            vtx_out=np.array([1]),
            n_per_hemi=2,
        )


def test_n_streamlines(endpoints):
    assert endpoints.n_streamlines == 4


# --- the smoothing itself --------------------------------------------------


def test_smoothing_produces_a_density(endpoints):
    """Symmetric, nonnegative, and summing to one over the whole matrix."""
    kernel = np.eye(2)
    density = smooth_endpoints(endpoints, kernel, kernel)

    assert density.shape == (4, 4)
    np.testing.assert_allclose(density, density.T)
    assert (density >= 0).all()
    assert density.sum() == pytest.approx(1.0)


def test_smoothing_with_an_identity_kernel_just_normalizes(endpoints):
    """With K = I the congruence is a no-op, so the result is A scaled to one."""
    a11, a22, a12 = endpoints.adjacency()
    total = a11.sum() + a22.sum() + 2 * a12.sum()
    density = smooth_endpoints(endpoints, np.eye(2), np.eye(2))

    np.testing.assert_allclose(density[:2, :2], a11 / total)
    np.testing.assert_allclose(density[2:, 2:], a22 / total)
    np.testing.assert_allclose(density[:2, 2:], a12 / total)


def test_smoothing_clips_negatives_from_spectral_truncation():
    """Truncation can make K A K negative; those entries become zero."""
    endpoints = Endpoints(
        surf_in=np.array([0], dtype=np.int8),
        surf_out=np.array([0], dtype=np.int8),
        vtx_in=np.array([0]),
        vtx_out=np.array([1]),
        n_per_hemi=2,
    )
    # A kernel with a negative off-diagonal drives one entry of K A K negative.
    kernel = np.array([[1.0, -0.6], [-0.6, 1.0]])
    density = smooth_endpoints(endpoints, kernel, np.eye(2))
    assert (density >= 0).all()
    assert density.sum() == pytest.approx(1.0)


def test_smoothing_rejects_a_mismatched_kernel(endpoints):
    with pytest.raises(ValueError, match="kernel_left has shape"):
        smooth_endpoints(endpoints, np.eye(3), np.eye(2))


def test_smoothing_rejects_empty_endpoints():
    empty = Endpoints(
        surf_in=np.array([], dtype=np.int8),
        surf_out=np.array([], dtype=np.int8),
        vtx_in=np.array([], dtype=np.int64),
        vtx_out=np.array([], dtype=np.int64),
        n_per_hemi=2,
    )
    with pytest.raises(ValueError, match="no streamlines to smooth"):
        smooth_endpoints(empty, np.eye(2), np.eye(2))


@pytest.fixture
def toy_endpoints():
    """Six streamlines over a toy grid of three vertices per hemisphere."""
    from sbci.smoothing import Endpoints

    return Endpoints(
        surf_in=np.array([0, 0, 0, 1, 1, 0], dtype=np.int8),
        surf_out=np.array([0, 1, 1, 1, 0, 0], dtype=np.int8),
        vtx_in=np.array([0, 1, 2, 0, 1, 2]),
        vtx_out=np.array([1, 0, 2, 2, 0, 1]),
        n_per_hemi=3,
    )


@pytest.fixture
def toy_basis():
    """A Laplace-Beltrami basis of that toy hemisphere: three vertices in a row, each of area 2.

    The path's Laplacian solved against the vertex areas, as laplace_beltrami_basis solves
    the grid's: a constant first mode at eigenvalue 0, the modes orthonormal under the
    areas, and a heat kernel that is nonnegative.
    """
    stiffness = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    eigenvalues, vectors = np.linalg.eigh(stiffness / 2.0)
    pair = (np.maximum(eigenvalues, 0.0), vectors / np.sqrt(2.0))
    return (pair, pair)


@pytest.fixture
def smoothable(sc_metadata, toy_endpoints):
    """A six-vertex SC connectome carrying those endpoints."""
    from sbci.connectome import ContinuousConnectome
    from sbci.grid import to_condensed

    rng = np.random.default_rng(1)
    matrix = rng.random((6, 6))
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 0.0)
    area = np.full(6, 2.0)
    matrix /= area @ matrix @ area
    return ContinuousConnectome(
        data=to_condensed(matrix).astype(np.float32),
        area=area,
        mask=np.ones(6, dtype=bool),
        metadata=metadata_template("sc", **{**sc_metadata.fields, "streamline_weighting": "none"}),
        endpoints=toy_endpoints,
    )


def test_smooth_says_so_when_the_file_carries_no_endpoints(connectome):
    """A file without the optional group cannot be re-smoothed."""
    assert not connectome.has_endpoints
    with pytest.raises(ValueError, match="no streamline endpoints"):
        connectome.smooth(kernel="rdk")


def test_a_functional_connectome_is_told_that_only_structural_ones_resmooth():
    """fc.smooth() told the user to build streamline endpoints for an FC file (C1-28)."""
    import sbci

    with pytest.raises(
        ValueError, match="Only a structural connectome can be re-smoothed"
    ) as caught:
        sbci.example("fc").smooth()
    assert "Endpoints.from_snapped" not in str(caught.value)


def test_rdk_on_a_toy_grid_needs_a_basis_of_its_own(smoothable):
    """The grid's own basis is built for the 5124-vertex grid; a toy grid says so plainly."""
    with pytest.raises(ValueError, match="6 vertex weights for a basis of 2562 \\+ 2562 vertices"):
        smoothable.smooth(kernel="rdk", bandwidth=1.0)


def test_the_basis_is_fetched_where_find_basis_looks(tmp_path, monkeypatch):
    """Fetched into $SBCI_LBO_DIR, or else ~/.cache/sbci/lbo: found there the next time."""
    from sbci import download, smoothing

    monkeypatch.delenv(smoothing.EIGENPAIR_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert download.basis_dir() in smoothing._candidate_basis_directories()
    monkeypatch.setenv(smoothing.EIGENPAIR_ENV, str(tmp_path / "lbo"))
    assert download.basis_dir() == tmp_path / "lbo"
    assert download.basis_dir() == smoothing._candidate_basis_directories()[0]
    assert tuple(download.BASIS_FILES) == smoothing.EIGENPAIR_FILES


def _pin(monkeypatch, folder, contents=(b"left basis", b"right basis")):
    """Write two small basis files and pin their sizes and digests, as BASIS_FILES pins the real."""
    import hashlib

    from sbci import download, smoothing

    pinned = {}
    for name, content in zip(smoothing.EIGENPAIR_FILES, contents, strict=True):
        (folder / name).write_bytes(content)
        pinned[name] = (len(content), hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(download, "BASIS_FILES", pinned)


def test_the_basis_is_found_from_the_environment_and_checked(tmp_path, monkeypatch):
    """Any two files of the right names were taken: two empty ones failed later in SciPy (A2-08)."""
    from sbci import smoothing
    from sbci.errors import MissingDataError

    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.delenv("SBCI_TOOLKIT", raising=False)
    monkeypatch.setenv(smoothing.EIGENPAIR_ENV, str(tmp_path))
    _pin(monkeypatch, tmp_path)
    assert smoothing.find_basis() == tmp_path
    (tmp_path / smoothing.EIGENPAIR_FILES[1]).write_bytes(b"right basiz")  # same size, damaged
    with pytest.raises(MissingDataError) as caught:
        smoothing.find_basis()
    message = str(caught.value)
    assert smoothing.EIGENPAIR_FILES[1] in message and "SHA-256" in message
    assert smoothing.EIGENPAIR_FILES[0] not in message, "the intact file is not blamed"
    assert "sbci basis --force" in message
    for name in smoothing.EIGENPAIR_FILES:
        (tmp_path / name).write_bytes(b"")
    with pytest.raises(MissingDataError, match="is 0 bytes, not the 10 of the SBCI_Toolkit's"):
        smoothing.find_basis()


def test_a_damaged_basis_is_skipped_for_an_intact_one_further_on(tmp_path, monkeypatch):
    from sbci import smoothing

    damaged, intact = tmp_path / "damaged", tmp_path / "home" / ".cache" / "sbci" / "lbo"
    for folder in (damaged, intact):
        folder.mkdir(parents=True)
    _pin(monkeypatch, intact)
    for name in smoothing.EIGENPAIR_FILES:
        (damaged / name).write_bytes(b"truncated")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SBCI_TOOLKIT", raising=False)
    monkeypatch.setenv(smoothing.EIGENPAIR_ENV, str(damaged))
    assert smoothing.find_basis() == intact


def test_a_basis_file_that_cannot_be_read_is_named_with_its_fix(tmp_path):
    """An empty file raised SciPy's 'Mat file appears to be truncated', naming nothing (A2-08)."""
    from sbci.errors import FormatError
    from sbci.smoothing import load_eigenpairs

    path = tmp_path / "EV_LBO_ds_ico4_L.mat"
    path.write_bytes(b"")
    with pytest.raises(FormatError) as caught:
        load_eigenpairs(path, "L")
    assert str(path) in str(caught.value) and "sbci basis --force" in str(caught.value)


def test_a_directory_missing_one_hemisphere_is_not_accepted(tmp_path, monkeypatch):
    """Half a basis is not a basis."""
    from sbci import smoothing

    (tmp_path / smoothing.EIGENPAIR_FILES[0]).write_bytes(b"")
    monkeypatch.setenv(smoothing.EIGENPAIR_ENV, str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    assert smoothing.find_basis() is None


def test_smooth_rejects_a_directory_without_the_basis(smoothable, tmp_path):
    with pytest.raises(FileNotFoundError, match="EV_LBO_ds_ico4_L.mat"):
        smoothable.smooth(kernel="rdk", eigenpairs=tmp_path)


@pytest.mark.parametrize("kernel", ["rdk", "matern"])
def test_smooth_returns_a_connectome_that_records_how_it_was_made(smoothable, toy_basis, kernel):
    """The whole path: endpoints in, a renormalized connectome out."""
    out = smoothable.smooth(kernel=kernel, bandwidth=1.0, eigenpairs=toy_basis)

    assert type(out) is type(smoothable)
    assert out.data.shape == smoothable.data.shape
    assert out.metadata["kernel"] == kernel
    assert out.metadata["bandwidth"] == 1.0
    assert out.endpoints is smoothable.endpoints
    # Unit mass is the package's convention, and re-smoothing must preserve it.
    area = np.asarray(out.area)
    assert area @ out.dense() @ area == pytest.approx(1.0, rel=1e-6)


def test_smooth_leaves_the_original_alone(smoothable, toy_basis):
    before = smoothable.data.copy()
    smoothable.smooth(kernel="rdk", bandwidth=1.0, eigenpairs=toy_basis)
    np.testing.assert_array_equal(smoothable.data, before)


def test_smooth_method_still_validates_its_arguments(connectome):
    with pytest.raises(ValueError, match="kernel must be one of"):
        connectome.smooth(kernel="gaussian")


# --- the spherical kernel concon applies -----------------------------------


def _fine_sphere(subdivisions=3):
    """A sphere fine enough to integrate on, with its vertex areas."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from sbci.alignment import voronoi_areas
    from test_alignment import icosphere

    vertices, faces = icosphere(subdivisions)
    return vertices, voronoi_areas(vertices, faces)


def test_the_kernel_does_not_integrate_to_one():
    """The kernel concon applies is not a probability density, and that is fine.

    A heat kernel integrates to 1 because its weight is (2l+1)/4pi and only the
    l=0 term survives integration. concon's weight is (2l+1)^(3/2)/sqrt(4pi), so
    its 33-term series, untruncated in angle, integrates to sqrt(4pi). That is
    not the kernel concon applies, though: it cuts the series off at 12.2
    degrees, dropping the negative lobes beyond, and the cut kernel integrates
    to 11.6 at sigma 0.005. No constant of either survives into a file: the
    pipeline divides the density by the streamline count, and the package
    normalizes it to unit mass.
    """
    from sbci.smoothing import spherical_heat_kernel

    pole = np.array([0.0, 0.0, 1.0])

    def integral(subdivisions, epsilon):
        vertices, areas = _fine_sphere(subdivisions)
        return float(spherical_heat_kernel(vertices @ pole, 0.005, epsilon=epsilon) @ areas)

    # The kernel spans about 12 degrees, so a coarse grid under-resolves it;
    # refining converges on sqrt(4 pi), which is what pins the series' constant.
    coarse, fine = integral(4, None), integral(5, None)
    target = np.sqrt(4 * np.pi)
    assert abs(fine - target) < abs(coarse - target), "should converge as the grid refines"
    assert fine == pytest.approx(target, rel=5e-3)
    # The kernel as applied, cut off and read through the binary's tables:
    # 11.5954 by quadrature in the cosine (A2-10)
    assert integral(5, 0.001) == pytest.approx(11.5954, rel=5e-3)


def test_the_kernel_falls_from_its_centre_then_stops_dead():
    """Inside the cutoff it decays; beyond it, it is exactly zero.

    The truncated series would ring past the cutoff, but concon never evaluates
    it there: compute_kernel.cpp only visits vertices within the cutoff angle,
    so the kernel has compact support and no ringing survives into the output.
    """
    from sbci.smoothing import kernel_cutoff, spherical_heat_kernel

    angles = np.linspace(0.0, np.pi, 400)
    values = spherical_heat_kernel(np.cos(angles), sigma=0.005)
    cutoff = kernel_cutoff(0.005)

    assert values[0] == values.max()
    inside = angles < cutoff
    assert np.all(np.diff(values[inside]) < 0), "should fall away from its centre"
    assert np.all(values[~inside] == 0.0), "and be exactly zero beyond the cutoff"
    assert values.min() >= 0.0, "so nothing rings negative"


def test_the_kernel_matches_the_binary_it_ports():
    """Pinned against c3_main itself, run on a single streamline.

    One streamline gives D(i,j) = K(theta_i) K(theta_j), so the output is the
    kernel. These are the values it produced at sigma 0.005, at the ico4 ring
    distances from a grid vertex; tests/reference/concon_probe.py regenerates
    them. The heat kernel this package used to implement misses them by 0.09.
    """
    from sbci.smoothing import spherical_heat_kernel

    angle = np.array([0.0, 3.962, 6.430, 7.930, 9.960, 11.890])
    measured = np.array([1.0000, 0.7441, 0.4442, 0.2751, 0.1035, 0.01016])

    values = spherical_heat_kernel(np.cos(np.radians(angle)), sigma=0.005)
    ratio = values / values[0]
    assert np.sqrt(np.mean((ratio - measured) ** 2)) < 0.0005

    # The absolute scale is right too, not just the shape: concon's own peak
    # was 270.10, read off a run whose endpoints sat exactly on vertices.
    assert float(values[0]) == pytest.approx(270.10, rel=1e-3)


def test_the_cutoff_reproduces_the_binarys_support():
    """The predicted cutoff has to keep exactly the vertices concon keeps.

    Measured from c3_main: 16, 31 and 61 vertices at sigma 0.0025, 0.005 and
    0.01, whose outermost rings sit at 7.932, 11.894 and 16.528 degrees.
    """
    from sbci.smoothing import kernel_cutoff

    for sigma, outermost, next_ring in [
        (0.0025, 7.932, 8.9),
        (0.005, 11.894, 12.61),
        (0.01, 16.528, 17.5),
    ]:
        cutoff = np.degrees(kernel_cutoff(sigma))
        assert outermost < cutoff < next_ring, (
            f"sigma {sigma}: cutoff {cutoff:.3f} must keep the ring at "
            f"{outermost} and drop the next"
        )


def test_a_wider_kernel_is_flatter():
    from sbci.smoothing import spherical_heat_kernel

    angles = np.linspace(0.0, np.pi, 60)
    narrow = spherical_heat_kernel(np.cos(angles), sigma=0.002)
    wide = spherical_heat_kernel(np.cos(angles), sigma=0.05)
    assert narrow.max() > wide.max()
    assert narrow.std() > wide.std()


def test_a_very_wide_kernel_goes_flat():
    """At a large bandwidth every term but l=0 dies, leaving a constant.

    That constant is 1/sqrt(4pi), not the heat kernel's 1/4pi, for the same
    reason the kernel does not integrate to one.
    """
    from sbci.smoothing import spherical_heat_kernel

    values = spherical_heat_kernel(np.cos(np.linspace(0, np.pi, 50)), sigma=5.0)
    np.testing.assert_allclose(values, 1 / np.sqrt(4 * np.pi), rtol=1e-3)


def test_the_truncation_is_thirty_two_and_is_part_of_the_definition():
    """The binary hardcodes 33 terms, and 33 is not converged.

    The pipeline passes --OPT_VAL_num_harm 33, but c3_main ignores it: it
    prints "num_harmonics now global constant (for speed)" and subject.cpp
    prints so at startup. Feeding the binary 9, 17, 25, 33, 49 and 65 gives a
    bit-identical kernel. The count was pinned at 33 by matching the binary's
    peak across bandwidths 0.00125 to 0.02, where 32 drifts to 4% and 34
    overshoots. Since the sum is not converged, a port that "improved" on it by
    summing further would stop reproducing the released cohorts.
    """
    from sbci.smoothing import NUM_HARMONICS, spherical_heat_kernel

    assert NUM_HARMONICS == 33, "33 terms, l = 0..32, whatever the flag says"

    cosine = np.cos(np.linspace(0, np.radians(12.0), 200))
    truncated = spherical_heat_kernel(cosine, sigma=0.005, epsilon=None)
    further = spherical_heat_kernel(cosine, sigma=0.005, harmonics=65, epsilon=None)
    assert np.abs(truncated - further).max() / truncated.max() > 1e-3, (
        "32 terms is not converged, so the truncation has to be matched"
    )


def test_endpoint_density_is_symmetric_and_hemisphere_respecting():
    from sbci.smoothing import endpoint_density

    vertices, _ = _fine_sphere(2)
    n = vertices.shape[0]
    vertex_hemisphere = np.zeros(n, dtype=int)
    vertex_hemisphere[n // 2 :] = 1

    rng = np.random.default_rng(0)
    points = vertices[rng.integers(0, n, 40)]
    hemisphere = vertex_hemisphere[rng.integers(0, n, 40)]

    density = endpoint_density(
        vertices,
        points,
        points,
        hemisphere,
        hemisphere,
        vertex_hemisphere,
        sigma=0.01,
    )
    assert density.shape == (n, n)
    np.testing.assert_allclose(density, density.T, atol=1e-12)
    assert density.min() >= 0


def test_endpoint_density_puts_mass_where_the_endpoints_are():
    """A single streamline should light up its own two endpoints."""
    from sbci.smoothing import endpoint_density

    vertices, _ = _fine_sphere(3)
    n = vertices.shape[0]
    vertex_hemisphere = np.zeros(n, dtype=int)

    first, second = 10, 400
    density = endpoint_density(
        vertices,
        vertices[[first]],
        vertices[[second]],
        np.array([0]),
        np.array([0]),
        vertex_hemisphere,
        sigma=0.005,
    )
    peak = np.unravel_index(np.argmax(density), density.shape)
    for index in peak:
        angle = np.degrees(np.arccos(np.clip(vertices[index] @ vertices[first], -1, 1)))
        other = np.degrees(np.arccos(np.clip(vertices[index] @ vertices[second], -1, 1)))
        assert min(angle, other) < 10.0, "the peak is nowhere near an endpoint"


def test_endpoint_density_validates_its_arguments():
    from sbci.smoothing import endpoint_density

    vertices, _ = _fine_sphere(1)
    n = vertices.shape[0]
    zero = np.zeros(n, dtype=int)
    with pytest.raises(ValueError, match="endpoints disagree"):
        endpoint_density(
            vertices,
            vertices[:3],
            vertices[:2],
            np.zeros(3, int),
            np.zeros(2, int),
            zero,
            sigma=0.01,
        )
    with pytest.raises(ValueError, match="normalize must be"):
        endpoint_density(
            vertices,
            vertices[:2],
            vertices[:2],
            np.zeros(2, int),
            np.zeros(2, int),
            zero,
            sigma=0.01,
            normalize="area",
        )


def test_reading_the_concon_endpoint_file(tmp_path):
    """The .tsv c3_main takes: a count header, then ten columns per streamline."""
    from sbci.smoothing import read_concon_endpoints

    path = tmp_path / "subject_xing_sphere_avg_coords.tsv"
    path.write_text(
        "#2\n"
        "0\t 1\t 1.0\t 0.0\t 0.0\t 0\t 0\t 0.0\t 1.0\t 0.0\n"
        "0\t 0\t 0.0\t 0.0\t 1.0\t 0\t 1\t 0.0\t 0.0\t -1.0\n"
    )
    points_in, points_out, hemisphere_in, hemisphere_out = read_concon_endpoints(path)

    assert points_in.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(points_in, axis=1), 1.0)
    # The file stores 1 - surface, so a 1 in the column means the left hemisphere.
    np.testing.assert_array_equal(hemisphere_in, [0, 1])
    np.testing.assert_array_equal(hemisphere_out, [1, 0])


def test_endpoint_positions_rebuilds_points_on_the_sphere():
    """The shk path needs continuous positions, and nothing else exercised it.

    Wiring smooth(kernel="shk") up the first time crashed here on a real file,
    because the triangle offset was read from an attribute Endpoints does not
    carry. Every unit test passed regardless: none of them had barycentric
    endpoints. This one does.
    """
    from sbci.smoothing import Endpoints, endpoint_positions
    from sbci.surface import load_surface

    sphere = load_surface("sphere")
    faces_per_hemi = sphere.faces.shape[0] // 2
    rng = np.random.default_rng(0)
    count = 64

    tri_in = rng.integers(0, faces_per_hemi, count)
    tri_out = rng.integers(0, faces_per_hemi, count)
    surf_in = rng.integers(0, 2, count).astype(np.int8)
    surf_out = rng.integers(0, 2, count).astype(np.int8)
    bary = rng.random((count, 3))
    bary /= bary.sum(axis=1, keepdims=True)

    def heaviest(tri, surf):  # the vertex a file stores: the corner of largest weight
        face = sphere.faces[tri + Endpoints.global_hemisphere_offset(surf, faces_per_hemi)]
        corner = face[np.arange(count), bary.argmax(axis=1)]
        return corner - Endpoints.global_hemisphere_offset(surf, sphere.n_vertices // 2)

    endpoints = Endpoints(
        surf_in=surf_in,
        surf_out=surf_out,
        vtx_in=heaviest(tri_in, surf_in),
        vtx_out=heaviest(tri_out, surf_out),
        tri_in=tri_in,
        tri_out=tri_out,
        bary_in=bary,
        bary_out=bary.copy(),
    )
    points_in, points_out = endpoint_positions(endpoints, surface=sphere)

    assert points_in.shape == (count, 3)
    np.testing.assert_allclose(np.linalg.norm(points_in, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(points_out, axis=1), 1.0, atol=1e-12)

    # A point must lie in its own triangle, so the nearest vertex of that
    # triangle is nearer than the triangle's own circumradius is wide.
    corners = sphere.vertices[
        sphere.faces[tri_in + Endpoints.global_hemisphere_offset(surf_in, faces_per_hemi)]
    ]
    corners = corners / np.linalg.norm(corners, axis=2, keepdims=True)
    inside = np.einsum("sj,skj->sk", points_in, corners).max(axis=1)
    assert np.all(inside > np.cos(np.radians(5.0))), "should sit inside its own triangle"


def test_endpoint_positions_refuses_a_file_without_them():
    from sbci.errors import MissingDataError
    from sbci.smoothing import Endpoints, endpoint_positions

    bare = Endpoints(
        surf_in=np.zeros(3, np.int8),
        surf_out=np.zeros(3, np.int8),
        vtx_in=np.zeros(3, int),
        vtx_out=np.zeros(3, int),
    )
    with pytest.raises(MissingDataError, match="barycentric"):
        endpoint_positions(bare)


@pytest.mark.parametrize(
    "shape", [(), (1,), (5,), (3, 4), (2, 3, 4)], ids=["scalar", "one", "vector", "2d", "3d"]
)
def test_the_kernel_preserves_the_shape_it_is_given(shape):
    """Including a scalar, which is a 0-d array and cannot be an `out=` target.

    The recurrence writes through `out=` buffers to avoid allocating 168 MB per
    term when smoothing a subject. numpy refuses a 0-d array there, so
    `spherical_heat_kernel(1.0, ...)` raised TypeError until the series learned
    to work on a flat view and reshape back.
    """
    from sbci.smoothing import spherical_heat_kernel

    cosine = np.full(shape, 0.999) if shape else 0.999
    assert np.shape(spherical_heat_kernel(cosine, 0.005)) == shape


def test_the_kernel_is_elementwise():
    """Evaluating together or one at a time has to give the same answer."""
    from sbci.smoothing import spherical_heat_kernel

    cosine = np.cos(np.radians(np.array([0.0, 4.0, 8.0, 12.0, 30.0])))
    together = spherical_heat_kernel(cosine, 0.005)
    apart = np.array([float(spherical_heat_kernel(c, 0.005)) for c in cosine])
    np.testing.assert_allclose(together, apart, rtol=0, atol=1e-12)


def test_resmoothing_masks_the_medial_wall_by_default_and_leaves_it_when_asked():
    """smooth() masks the medial wall unless told not to (M6, SPEC_QUESTIONS.md item 14).

    The format requires a stored file to carry nothing on the medial wall, and
    every released file was smoothed with the wall masked; neither reference
    masks. The default used to follow the references, so re-smoothing a
    released file neither gave it back nor passed ``sbci validate``. It now
    follows the format; ``mask_medial_wall=False`` keeps what the kernel put
    there, which is what a comparison with the references wants.
    """
    from sbci.grid import to_dense
    from sbci.smoothing import _finish

    n = 8
    rng = np.random.default_rng(0)
    dense = rng.random((n, n))
    dense = dense + dense.T
    mask = np.ones(n, dtype=bool)
    mask[[2, 5]] = False

    class Fake:
        def __init__(self, data, area, mask, metadata, coords, endpoints):
            self.data, self.area, self.mask = data, area, mask
            self.metadata, self.coords, self.endpoints = metadata, coords, endpoints

    connectome = Fake(None, np.full(n, 1.0 / n), mask, metadata_template("sc"), None, None)
    masked = to_dense(_finish(connectome, dense.copy(), "shk", 0.005).data, n)
    kept = to_dense(_finish(connectome, dense.copy(), "shk", 0.005, mask_medial_wall=False).data, n)
    assert float(np.abs(masked[~mask]).sum()) == 0.0, "the default masks the wall"
    assert float(np.abs(kept[~mask]).sum()) > 0.0, "and False keeps what the kernel gave it"
    for result in (masked, kept):
        assert float(
            connectome.area @ result.astype(np.float64) @ connectome.area
        ) == pytest.approx(1.0, rel=1e-6)


def test_resmoothing_with_the_defaults_gives_the_file_back_and_validates(tmp_path):
    """Re-smoothed with the defaults, a released-style file kept a wall and failed validate (M6)."""
    import sbci
    from sbci.validate import validate_file

    sc = sbci.example()
    again = sc.smooth()
    assert np.corrcoef(np.asarray(again.data), np.asarray(sc.data))[0, 1] > 0.999999
    np.testing.assert_array_equal(np.asarray(again.data), np.asarray(sc.data))
    failed = [
        check.name
        for check in validate_file(again.save(tmp_path / "sub-synth01_sc.h5"))
        if not check.passed
    ]
    assert failed == []


def test_the_quantized_kernel_reproduces_the_binarys_peak_entry_exactly():
    """c3_main reads the kernel through two truncation-indexed lookup tables.

    On a single streamline its peak entry is K(dot_1) * K(dot_2) with one dot
    rounding into the exact x = 1.0 bucket and the other one truncation step
    below. The measured peaks were 270.1035 at sigma 0.005 and 1005.4661 at
    0.00125; the geometric mean of the two buckets has to give them back. The
    exact series cannot: it is 0.06% high at the peak, which is the 0.14%
    amplitude offset the densities used to carry.
    """
    from sbci.smoothing import concon_kernel_table

    for sigma, measured in ((0.005, 270.1035), (0.00125, 1005.4661), (0.02, 48.3245)):
        table = concon_kernel_table(sigma)
        peak = float(np.sqrt(table(1.0) * table(1.0 - 1e-12)))
        assert peak == pytest.approx(measured, rel=2e-5), (sigma, peak, measured)


def test_quantized_and_exact_kernels_differ_only_by_the_table_bias():
    from sbci.smoothing import spherical_heat_kernel

    cosine = np.cos(np.radians(np.linspace(0.0, 11.5, 50)))
    quantized = spherical_heat_kernel(cosine, 0.005)
    exact = spherical_heat_kernel(cosine, 0.005, quantized=False)
    ratio = quantized / exact
    assert np.all(ratio <= 1.0 + 1e-12), "truncation only ever rounds the kernel down"
    assert ratio.min() > 0.995, "and by well under one percent"


def test_quantized_and_exact_share_the_cutoff():
    from sbci.smoothing import concon_kernel_table, kernel_cutoff

    for sigma in (0.0025, 0.005, 0.01):
        assert concon_kernel_table(sigma).cutoff == pytest.approx(kernel_cutoff(sigma), abs=2e-4)


def test_sparse_density_matches_dense():
    """The KD-tree path has to give the dense evaluation's answer, not an approximation."""
    from sbci.grid import hemisphere_labels
    from sbci.smoothing import endpoint_density
    from sbci.surface import load_surface

    sphere = load_surface("sphere")
    vertices = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
    hemi = hemisphere_labels(vertices.shape[0])
    rng = np.random.default_rng(0)
    count = 600
    p_in = rng.normal(size=(count, 3))
    p_in /= np.linalg.norm(p_in, axis=1, keepdims=True)
    p_out = rng.normal(size=(count, 3))
    p_out /= np.linalg.norm(p_out, axis=1, keepdims=True)
    h_in = rng.integers(0, 2, count)
    h_out = rng.integers(0, 2, count)

    dense = endpoint_density(
        vertices, p_in, p_out, h_in, h_out, hemi, 0.005, method="dense", block=256
    )
    sparse = endpoint_density(
        vertices, p_in, p_out, h_in, h_out, hemi, 0.005, method="sparse", block=256
    )
    assert dense.max() > 0
    np.testing.assert_allclose(sparse, dense, rtol=0, atol=1e-9 * dense.max())


def test_the_south_pole_takes_its_mass_which_c3_main_never_writes():
    """The vertex at (0, 0, -1) gets what the kernel puts there.

    ``c3_main`` looks the vertices near an endpoint up in a quad tree over the
    polar angle, clipping the range at pi, and the tree's range test is
    half-open, so the vertex at exactly pi -- vertex 11 of each hemisphere -- is
    never found, and every connectome the binary made is empty there
    (PORTING.md item 6). That is the reference's error, not this package's.
    """
    from sbci.grid import hemisphere_labels
    from sbci.smoothing import endpoint_density
    from sbci.surface import load_surface

    sphere = load_surface("sphere")
    vertices = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
    hemi = hemisphere_labels(vertices.shape[0])
    half = vertices.shape[0] // 2
    poles = np.flatnonzero(vertices[:, 2] < -0.9999)
    assert poles.tolist() == [11, half + 11]

    tilt = np.radians(1.0)  # each streamline starts a degree from the south pole
    start = np.array([[np.sin(tilt), 0.0, -np.cos(tilt)]] * 2)
    end = np.array([[1.0, 0.0, 0.0]] * 2)
    side = np.array([0, 1])
    density = endpoint_density(vertices, start, end, side, side, hemi, 0.005)
    rows = density.sum(axis=1)
    for pole, hemisphere in zip(poles, (0, 1), strict=True):
        near = (hemi == hemisphere) & (vertices @ start[0] > np.cos(np.radians(10.0)))
        assert rows[pole] > 0
        assert np.flatnonzero(near)[np.argmax(rows[near])] == pole, "the pole is the nearest vertex"


def test_progress_callback_reports_every_block():
    from sbci.grid import hemisphere_labels
    from sbci.smoothing import endpoint_density
    from sbci.surface import load_surface

    sphere = load_surface("sphere")
    vertices = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
    hemi = hemisphere_labels(vertices.shape[0])
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(100, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    seen = []
    endpoint_density(
        vertices,
        pts,
        pts[::-1],
        np.zeros(100, int),
        np.zeros(100, int),
        hemi,
        0.005,
        block=30,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(30, 100), (60, 100), (90, 100), (100, 100)]


def test_mask_medial_wall_option_zeroes_the_wall_and_keeps_unit_mass():
    """The caller can choose the format's side of SPEC_QUESTIONS item 14."""
    from sbci.grid import to_dense
    from sbci.smoothing import _finish

    n = 8
    rng = np.random.default_rng(0)
    dense = rng.random((n, n))
    dense = dense + dense.T
    mask = np.ones(n, dtype=bool)
    mask[[2, 5]] = False

    class Fake:
        def __init__(self, data, area, mask, metadata, coords, endpoints):
            self.data, self.area, self.mask = data, area, mask
            self.metadata, self.coords, self.endpoints = metadata, coords, endpoints

    connectome = Fake(None, np.full(n, 1.0 / n), mask, metadata_template("sc"), None, None)
    out = to_dense(_finish(connectome, dense.copy(), "shk", 0.005, mask_medial_wall=True).data, n)
    assert float(np.abs(out[~mask]).sum()) == 0.0 and float(np.abs(out[:, ~mask]).sum()) == 0.0
    assert float(connectome.area @ out.astype(np.float64) @ connectome.area) == pytest.approx(
        1.0, rel=1e-6
    )


def test_final_threshold_matches_compute_kernel_cpp():
    """`if(temp > final_thold)` on the per-streamline scale, nothing else."""
    from sbci.smoothing import FINAL_THRESHOLD, apply_final_threshold

    assert FINAL_THRESHOLD == 1e-9, "--final_thold 0.000000001"
    n = 1_000_000
    density = np.array(
        [[0.0, 2e-9 * n, 1e-9 * n], [2e-9 * n, 0.0, 0.5e-9 * n], [1e-9 * n, 0.5e-9 * n, 0.0]]
    )
    out = apply_final_threshold(density.copy(), n)
    assert out[0, 1] == 2e-9 * n, "above the threshold survives"
    assert out[0, 2] == 0.0, "exactly at the threshold is dropped: the test is strictly greater"
    assert out[1, 2] == 0.0, "below is dropped"


# --- validation and the default path, end to end -----------------------------


def test_from_global_refuses_inconsistent_triangles():
    from sbci.smoothing import Endpoints

    vertex_in, vertex_out = np.array([1, 3000]), np.array([2, 3001])
    good = Endpoints.from_global(
        vertex_in,
        vertex_out,
        triangle_in=[5, 5125],
        triangle_out=[6, 5126],
        barycentric_in=np.full((2, 3), 1 / 3),
        barycentric_out=np.full((2, 3), 1 / 3),
    )
    np.testing.assert_array_equal(good.tri_in, [5, 5])
    with pytest.raises(ValueError, match="other hemisphere"):
        Endpoints.from_global(
            vertex_in,
            vertex_out,
            triangle_in=[5125, 5],
            triangle_out=[6, 5126],
            barycentric_in=np.full((2, 3), 1 / 3),
            barycentric_out=np.full((2, 3), 1 / 3),
        )
    with pytest.raises(ValueError, match="out of range"):
        Endpoints.from_global(
            vertex_in,
            vertex_out,
            triangle_in=[5, 20000],
            triangle_out=[6, 5126],
            barycentric_in=np.full((2, 3), 1 / 3),
            barycentric_out=np.full((2, 3), 1 / 3),
        )
    with pytest.raises(ValueError, match="positions need all"):
        Endpoints.from_global(vertex_in, vertex_out, triangle_in=[5, 5125])


def test_endpoints_check_their_own_consistency():
    from sbci.smoothing import Endpoints

    with pytest.raises(ValueError, match="entries"):
        Endpoints(surf_in=[0, 0], surf_out=[0], vtx_in=[1, 2], vtx_out=[3, 4])
    with pytest.raises(ValueError, match="0 .* or 1"):
        Endpoints(surf_in=[0, 2], surf_out=[0, 0], vtx_in=[1, 2], vtx_out=[3, 4])
    with pytest.raises(ValueError, match="positions need all"):
        Endpoints(surf_in=[0], surf_out=[0], vtx_in=[1], vtx_out=[3], tri_in=[2])


def test_endpoint_density_validates_its_inputs():
    from sbci.smoothing import endpoint_density

    vertices = np.eye(3)
    points = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="one entry per streamline"):
        endpoint_density(vertices, points, points, [0, 0], [0], [0, 0, 0], sigma=0.005)
    with pytest.raises(ValueError, match="finite unit vectors"):
        endpoint_density(2 * vertices, points, points, [0], [0], [0, 0, 0], sigma=0.005)


def test_the_kernel_without_a_cutoff_is_the_closed_form_series():
    """``epsilon=None`` studies the series itself; the table is only meaningful inside its cutoff.

    Inside the cutoff the table tracks the series to a few tenths of a percent
    (the truncation bias grows with the angle); beyond it the series goes
    negative while the table holds the binary's 1e-20 floor.
    """
    from sbci.smoothing import concon_kernel_table, spherical_heat_kernel

    cosines = np.cos(np.radians(np.array([0.0, 5.0, 11.0, 20.0, 40.0])))
    table = concon_kernel_table(0.005)
    truncated = spherical_heat_kernel(cosines, 0.005)
    untruncated = spherical_heat_kernel(cosines, 0.005, epsilon=None)
    assert truncated[-1] == 0.0 and untruncated[-1] != 0.0
    # No cutoff means the closed-form series, not the table read past its cutoff:
    # the table clamps the negative lobes to 1e-20, so it is a floor out there.
    np.testing.assert_array_equal(
        untruncated, spherical_heat_kernel(cosines, 0.005, epsilon=None, quantized=False)
    )
    assert untruncated[-1] < 0.0 < table(cosines, truncate=False)[-1] <= 1e-19
    np.testing.assert_allclose(truncated[:3], untruncated[:3], rtol=3e-3)


def test_smooth_shk_end_to_end_from_barycentric_endpoints():
    """The default kernel's whole path on the real grid: locate, spread, threshold, finish."""
    import sbci
    from sbci.alignment import MeshQuery
    from sbci.grid import to_dense
    from sbci.smoothing import (
        Endpoints,
        apply_final_threshold,
        endpoint_density,
        endpoint_positions,
    )
    from sbci.surface import load_surface

    rng = np.random.default_rng(21)
    sphere = load_surface("sphere")
    half, faces_per_hemi = sphere.n_vertices // 2, len(sphere.faces) // 2
    n = 300
    points = rng.normal(size=(2 * n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    hemispheres = np.r_[np.zeros(n, int), np.ones(n, int)]
    vertex = np.empty(2 * n, dtype=np.int64)
    triangle = np.empty(2 * n, dtype=np.int64)
    bary = np.empty((2 * n, 3))
    for side, name in ((0, "L"), (1, "R")):
        part = sphere.hemisphere(name)
        # on the unit sphere, where the points are: located on the bundled sphere, of
        # radius 100, they landed a median 3.4 degrees from where they were drawn (A2-10)
        unit = part.vertices / np.linalg.norm(part.vertices, axis=1, keepdims=True)
        w, i, f = MeshQuery(unit, part.faces).query_faces(points[hemispheres == side])
        vertex[hemispheres == side] = i[np.arange(len(i)), np.argmax(w, axis=1)] + side * half
        triangle[hemispheres == side] = f + side * faces_per_hemi
        bary[hemispheres == side] = w
    endpoints = Endpoints.from_global(
        vertex[:n],
        vertex[n:],
        triangle_in=triangle[:n],
        triangle_out=triangle[n:],
        barycentric_in=bary[:n],
        barycentric_out=bary[n:],
    )

    cc = sbci.example("sc")
    cc.endpoints = endpoints
    rebuilt = np.concatenate(endpoint_positions(endpoints))
    drawn = np.concatenate([points[:n], points[n:]])
    offset = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", rebuilt, drawn), -1, 1)))
    assert offset.max() < 0.01, "each endpoint is where it was drawn"
    smoothed = cc.smooth(kernel="shk", mask_medial_wall=False)
    dense = smoothed.dense().astype(np.float64)
    assert smoothed.metadata["kernel"] == "shk" and smoothed.metadata["bandwidth"] == 0.005
    assert float(cc.area @ dense @ cc.area) == pytest.approx(1.0, rel=1e-6)
    assert (dense >= 0).all() and np.allclose(dense, dense.T)

    # the same density by the dense path, finished the same way
    p_in, p_out = endpoint_positions(endpoints)
    vertices = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
    labels = np.r_[np.zeros(half, np.int8), np.ones(half, np.int8)]
    check = endpoint_density(
        vertices,
        p_in,
        p_out,
        endpoints.surf_in,
        endpoints.surf_out,
        labels,
        sigma=0.005,
        method="dense",
    )
    apply_final_threshold(check, n)
    np.fill_diagonal(check, 0.0)
    check /= cc.area @ check @ cc.area
    np.testing.assert_allclose(
        to_dense(smoothed.data.astype(np.float64)), check, rtol=1e-5, atol=1e-12
    )


def test_the_density_refuses_nan_positions_and_unknown_hemisphere_flags():
    from sbci.smoothing import endpoint_density

    vertices = np.eye(3)
    points = np.array([[1.0, 0.0, 0.0]])
    good = dict(
        vertices=vertices,
        hemisphere_in=np.array([0]),
        hemisphere_out=np.array([0]),
        vertex_hemisphere=np.zeros(3, dtype=int),
        sigma=0.5,
    )
    with pytest.raises(ValueError, match="finite unit vectors"):
        endpoint_density(points_in=np.array([[np.nan, 0.0, 0.0]]), points_out=points, **good)
    with pytest.raises(ValueError, match="match no vertex hemisphere"):
        endpoint_density(
            points_in=points,
            points_out=points,
            **{**good, "hemisphere_out": np.array([1])},
        )


def test_a_triangle_index_beyond_its_hemisphere_is_refused():
    from sbci.smoothing import Endpoints

    with pytest.raises(ValueError, match="within a hemisphere"):
        Endpoints(
            surf_in=np.array([0]),
            surf_out=np.array([0]),
            vtx_in=np.array([0]),
            vtx_out=np.array([1]),
            tri_in=np.array([6000]),
            tri_out=np.array([0]),
            bary_in=np.array([[1.0, 0.0, 0.0]]),
            bary_out=np.array([[1.0, 0.0, 0.0]]),
        )


def test_the_default_kernel_needs_the_ico4_grid(smoothable, toy_endpoints, toy_basis):
    from dataclasses import replace

    corner = np.tile([1.0, 0.0, 0.0], (6, 1))
    smoothable.endpoints = replace(  # positioned, so the grid is the next thing checked
        toy_endpoints,
        tri_in=np.zeros(6, dtype=int),
        tri_out=np.zeros(6, dtype=int),
        bary_in=corner,
        bary_out=corner,
    )
    with pytest.raises(ValueError, match="shk smoothing needs"):
        smoothable.smooth(kernel="shk", eigenpairs=toy_basis)


def test_an_index_past_its_hemisphere_is_refused_rather_than_moved_to_the_other():
    """Left vertex 2600 was stored as whole-grid 2600 and read back as right vertex 38."""
    from sbci.smoothing import Endpoints

    with pytest.raises(ValueError, match=r"vtx_in out of range for 2562 .*\[2600, 2600\]"):
        Endpoints(surf_in=[0], surf_out=[0], vtx_in=[2600], vtx_out=[5])
    with pytest.raises(ValueError, match=r"vtx_out out of range .*\[-1, -1\]"):
        Endpoints(surf_in=[1], surf_out=[1], vtx_in=[3], vtx_out=[-1])
    weights = np.full((1, 3), 1 / 3)
    with pytest.raises(ValueError, match=r"tri_out: triangle indices must lie in 0..5119"):
        Endpoints([0], [0], [3], [5], tri_in=[7], tri_out=[5120], bary_in=weights, bary_out=weights)
    kept = Endpoints([0, 1], [1, 0], [2561, 0], [0, 2561])
    assert kept.global_vertex_in.tolist() == [2561, 2562]


# --- the review of 11 October 2026 -------------------------------------------------


def test_hemisphere_flags_and_vertices_are_checked_before_they_are_stored():
    """int8 wrapped 256 to 0 and truncated 1.9 to 1; a vertex 5.9 was stored as 5."""
    from sbci.smoothing import Endpoints

    for wrong in ([256], [257], [-255], [1.9]):
        with pytest.raises(ValueError, match="0 \\(left\\) or 1 \\(right\\)"):
            Endpoints(surf_in=wrong, surf_out=[0], vtx_in=[5], vtx_out=[5])
    with pytest.raises(ValueError, match="whole numbers"):
        Endpoints(surf_in=[0], surf_out=[0], vtx_in=[5.9], vtx_out=[5])
    assert Endpoints(surf_in=[1.0], surf_out=[0], vtx_in=[5.0], vtx_out=[5]).vtx_in.tolist() == [5]


def test_smooth_refuses_a_functional_connectome():
    """Endpoints on an FC connectome were smoothed into a unit-mass density labelled 'fc'."""
    import sbci

    sc, fc = sbci.example(), sbci.example("fc")
    mixed = sbci.ContinuousConnectome(
        fc.data, fc.area, fc.mask, fc.metadata, endpoints=sc.endpoints
    )
    with pytest.raises(ValueError, match="structural connectome"):
        mixed.smooth(kernel="shk")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("surf_in", 1.9, "0 \\(left\\) or 1"),
        ("surf_in", 256.0, "0 \\(left\\) or 1"),
        ("vtx_in", 2.5, "whole"),
    ],
)
def test_from_matlab_hands_the_constructor_the_values_as_stored(tmp_path, field, value, message):
    """It cast first: 1.9 became hemisphere 1, 256 hemisphere 0, vertex 2.5 vertex 1."""
    import scipy.io

    values = {"surf_in": [[0.0], [1.0]], "surf_out": [[0.0], [1.0]], "vtx_in": [[1.0], [2.0]]}
    values["vtx_out"] = [[1.0], [2.0]]
    values[field] = [[0.0], [value]] if field.startswith("surf") else [[1.0], [value]]
    path = tmp_path / "mesh_intersections.mat"
    scipy.io.savemat(path, values)
    with pytest.raises(ValueError, match=message):
        Endpoints.from_matlab(path, n_per_hemi=10)


# --- the review of 20 October 2026 -------------------------------------------------


def test_nu_shapes_the_matern_kernel_and_is_recorded(smoothable, toy_basis, tmp_path):
    """smooth() always used nu = 3 and recorded none, so files at two nu looked alike."""
    from sbci import ContinuousConnectome

    def matern(**kwargs):
        return smoothable.smooth(kernel="matern", bandwidth=1.0, eigenpairs=toy_basis, **kwargs)

    one, three = matern(nu=1), matern()
    assert not np.allclose(one.data, three.data)
    np.testing.assert_array_equal(three.data, matern(nu=3).data)
    assert one.metadata["nu"] == 1.0 and three.metadata["nu"] == 3.0
    back = ContinuousConnectome.load(one.save(tmp_path / "sub-toy_sc.h5"))
    assert back.metadata["nu"] == 1.0
    # the Matern kernel's alone: another kernel drops it and refuses one
    assert (
        "nu" not in back.smooth(kernel="rdk", bandwidth=1.0, eigenpairs=toy_basis).metadata.fields
    )
    with pytest.raises(ValueError, match="nu is the Matern kernel's smoothness"):
        smoothable.smooth(kernel="rdk", bandwidth=1.0, eigenpairs=toy_basis, nu=2)
    for bad, error in (
        (0, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
    ):
        with pytest.raises(error, match="nu "):
            matern(nu=bad)


@pytest.mark.parametrize("bad", [True, np.True_, "0.005", b"0.005"])
def test_smooth_refuses_a_bandwidth_that_is_not_a_number(smoothable, bad):
    """float() read True as sigma 1 and "0.005" as 0.005, and the file recorded them so."""
    with pytest.raises(TypeError, match="bandwidth takes a number"):
        smoothable.smooth(kernel="shk", bandwidth=bad)


def test_a_float32_bandwidth_is_the_decimal_it_prints_as(smoothable, toy_basis):
    """np.float32(0.005) was recorded as 0.004999999888 and refused beside files at 0.005."""
    single = smoothable.smooth(kernel="rdk", bandwidth=np.float32(0.005), eigenpairs=toy_basis)
    double = smoothable.smooth(kernel="rdk", bandwidth=0.005, eigenpairs=toy_basis)
    assert single.metadata["bandwidth"] == 0.005
    np.testing.assert_array_equal(single.data, double.data)


def test_the_sparse_block_shrinks_for_a_wide_kernel_and_not_at_the_default(monkeypatch):
    """At sigma 0.5 a block of 65,536 endpoints made 3.0 GB of pair arrays a side, unbounded."""
    from sbci import smoothing
    from sbci.smoothing import PAIR_BYTES, SPARSE_BLOCK, _sparse_block, concon_kernel_table

    assert _sparse_block(concon_kernel_table(0.005).cutoff, 2562) == SPARSE_BLOCK
    cutoff = concon_kernel_table(0.5).cutoff
    pairs = PAIR_BYTES * 2562 * (1 - np.cos(cutoff)) / 2  # bytes an endpoint's pairs take
    block = _sparse_block(cutoff, 2562)
    assert block * pairs <= smoothing.SPARSE_PAIR_BUDGET < (block + 1) * pairs
    assert block < SPARSE_BLOCK / 2
    # and endpoint_density takes it: a budget of ten and a half endpoints makes blocks of ten
    monkeypatch.setattr(smoothing, "SPARSE_PAIR_BUDGET", 10.5 * pairs)
    from sbci.grid import hemisphere_labels
    from sbci.surface import load_surface

    vertices = load_surface("sphere").vertices
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    seen = []
    smoothing.endpoint_density(
        vertices,
        vertices[:25],
        vertices[25:50],
        np.zeros(25, int),
        np.zeros(25, int),
        hemisphere_labels(),
        0.5,
        progress=lambda done, total: seen.append(done),
    )
    assert seen == [10, 20, 25]


def test_a_nan_cosine_gives_a_nan_kernel():
    """The index cast mapped NaN onto the table: spherical_heat_kernel(nan, 0.005) was 0.2423."""
    from sbci.smoothing import concon_kernel_table, spherical_heat_kernel

    values = spherical_heat_kernel(np.array([np.nan, 1.0]), 0.005)
    assert np.isnan(values[0]) and values[1] == spherical_heat_kernel(1.0, 0.005)
    assert np.isnan(spherical_heat_kernel(np.nan, 0.005))
    assert np.isnan(concon_kernel_table(0.005)(np.nan, truncate=False))


@pytest.mark.parametrize("bad", [0.0, -0.005, np.nan, np.inf])
def test_the_spherical_kernel_needs_a_positive_finite_sigma(bad):
    """A NaN sigma gave a 180-degree cutoff and NaNs; a negative one weighed high degrees up."""
    from sbci.smoothing import (
        concon_kernel_table,
        endpoint_density,
        kernel_cutoff,
        spherical_heat_kernel,
    )

    point = np.array([[1.0, 0.0, 0.0]])
    for call in (
        lambda: spherical_heat_kernel(1.0, bad),
        lambda: kernel_cutoff(bad),
        lambda: concon_kernel_table(bad),
        lambda: endpoint_density(np.eye(3), point, point, [0], [0], [0, 0, 0], sigma=bad),
    ):
        with pytest.raises(ValueError, match="sigma must be positive and finite"):
            call()


@pytest.mark.parametrize("block", [0, -1, 2.5, True, "64"])
@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_a_block_must_be_a_positive_whole_number(block, method):
    """block=-1 made an empty range of blocks, and the density came back all zeros."""
    from sbci.smoothing import endpoint_density

    point = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="block must be a positive whole number"):
        endpoint_density(
            np.eye(3), point, point, [0], [0], [0, 0, 0], 0.005, block=block, method=method
        )


def test_barycentric_rows_must_sum_to_one_and_may_dip_below_zero():
    """A row of zeros put its endpoint at NaN; a row summing to 2 counted twice in ConSEAL."""

    def ends(weights):
        corner = [[1.0, 0.0, 0.0]]
        return Endpoints(
            [0], [0], [0], [1], tri_in=[0], tri_out=[0], bary_in=weights, bary_out=corner
        )

    for bad in ([[0.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]], [[0.5, 0.5, 0.0005]]):
        with pytest.raises(ValueError, match="bary_in holds 1 row\\(s\\) that do not sum to 1"):
            ends(bad)
    assert ends([[1.0075, -0.0075, 0.0]]).bary_in[0, 1] == -0.0075  # just outside its triangle
    ends(np.array([[0.1, 0.2, 0.7]], dtype=np.float32))  # as a file stores them
    with pytest.raises(ValueError, match="barycentric_out holds 1 row"):
        Endpoints.from_global(
            [0],
            [1],
            triangle_in=[0],
            triangle_out=[0],
            barycentric_in=[[1.0, 0.0, 0.0]],
            barycentric_out=[[0.0, 0.0, 0.0]],
        )


def test_endpoint_positions_refuse_a_vertex_beyond_one_edge_of_its_triangle():
    """Triangles stored against another face order rebuilt 85 degrees from their vertex, unseen."""
    import sbci
    from sbci.smoothing import _vertex_offsets, endpoint_positions

    sc = sbci.example()
    ends = sc.endpoints
    _, _, offsets, edge = _vertex_offsets(ends)
    assert 4.5 < edge < 5.0 and offsets.max() < 0.6 * edge
    endpoint_positions(ends)
    ends.tri_in = 5119 - ends.tri_in  # the same triangles, in another order
    with pytest.raises(ValueError, match="farther than one edge of the grid"):
        endpoint_positions(ends)
    holder = sbci.ContinuousConnectome(sc.data, sc.area, sc.mask, sc.metadata, endpoints=ends)
    with pytest.raises(ValueError, match="farther than one edge of the grid"):
        holder.smooth(kernel="shk")


# --- the review of 23 November 2026 -------------------------------------------------


def test_rdk_builds_the_grids_own_basis_with_nothing_to_fetch(monkeypatch):
    """The toolkit's basis was fetched, 100 MB, and was not orthonormal on the grid (A2-02).

    The grid's own is the bundled white surface's Laplace-Beltrami operator, solved in
    full: orthonormal under its vertex areas, with a constant first mode at 0.
    """
    from sbci import download, smoothing
    from sbci.surface import load_surface

    def refuse(*args, **kwargs):
        raise AssertionError("the toolkit's basis was looked for")

    monkeypatch.setattr(smoothing, "find_basis", refuse)
    monkeypatch.setattr(download, "fetch_basis", refuse)
    for side in "LR":
        eigenvalues, eigenvectors, areas = smoothing.laplace_beltrami_basis(side)
        assert eigenvalues.shape == (2562,) and eigenvalues[0] == 0.0
        assert (np.diff(eigenvalues) >= 0).all() and eigenvalues[1] > 0
        gram = eigenvectors.T @ (areas[:, None] * eigenvectors)
        assert np.abs(gram - np.eye(2562)).max() < 1e-10
        np.testing.assert_allclose(eigenvectors[:, 0], eigenvectors[0, 0], rtol=1e-10)
        assert not eigenvalues.flags.writeable, "shared between calls, so read-only"
    white = load_surface("white").hemisphere("L")
    stiffness, _ = smoothing.laplace_beltrami_operator(white.vertices, white.faces)
    assert (stiffness - np.diag(np.diag(stiffness)) <= 1e-12).all(), "every edge weight >= 0"


def test_the_rdk_kernel_is_nonnegative_local_and_of_unit_mass_per_endpoint():
    """At the reference's bandwidth 38% of the kernel's weight was negative, 55% beyond 20 degrees.

    And under the file's weights an endpoint's kernel integrated to 0.64-1.51, by where
    it landed (A2-02, A2-03). The kernel smooth() applies is checked here at that
    bandwidth and at its own default.
    """
    from sbci.io.cifti import vertex_areas
    from sbci.smoothing import DEFAULT_KAPPA, kernel_matrices
    from sbci.surface import load_surface

    area = np.asarray(vertex_areas()[1], dtype=np.float64)
    sphere = load_surface("sphere").vertices[:2562]
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip(sphere @ sphere.T, -1.0, 1.0)))
    for kappa, far in ((1.9483, 1e-4), (DEFAULT_KAPPA["rdk"], 1e-2)):
        left, right = kernel_matrices("rdk", kappa, area=area)
        np.testing.assert_allclose(area[:2562] @ left, 1.0, rtol=1e-12)
        np.testing.assert_allclose(area[2562:] @ right, 1.0, rtol=1e-12)
        weighted = area[:2562, None] * left
        assert weighted.min() >= -1e-12 * weighted.max(), "nonnegative, to rounding"
        beyond = (weighted * (angle > 20.0)).sum(axis=0)
        assert np.median(beyond) < far, f"kappa {kappa}: {np.median(beyond):.2e} beyond 20 degrees"


def test_every_streamline_carries_the_same_mass_with_rdk(monkeypatch):
    """Under the shipped basis a streamline's mass went as g(p) g(q), 0.64-1.51 an end (A2-03)."""
    import sbci
    from sbci import download, smoothing
    from sbci.smoothing import DEFAULT_KAPPA, Endpoints
    from sbci.surface import load_surface

    monkeypatch.setattr(smoothing, "find_basis", lambda: None)
    monkeypatch.setattr(download, "fetch_basis", lambda **kwargs: None)
    sc = sbci.example()
    sphere = load_surface("sphere").vertices[:2562]
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    # two left-hemisphere streamlines, their four ends far from each other and from the wall
    cortex = np.flatnonzero(sc.mask[:2562])
    wall = sphere[~sc.mask[:2562]]
    clear = cortex[(sphere[cortex] @ wall.T).max(axis=1) < np.cos(np.radians(25.0))]
    rng = np.random.default_rng(4)
    while True:
        ends = rng.choice(clear, 4, replace=False)
        gaps = sphere[ends] @ sphere[ends].T
        if (gaps[np.triu_indices(4, 1)] < np.cos(np.radians(60.0))).all():
            break
    sc.endpoints = Endpoints(surf_in=[0, 0], surf_out=[0, 0], vtx_in=ends[:2], vtx_out=ends[2:])
    out = sc.smooth(kernel="rdk")
    assert out.metadata["bandwidth"] == DEFAULT_KAPPA["rdk"]
    assert "the grid's own Laplace-Beltrami eigenpairs" in out.metadata["provenance"]
    dense = out.dense(np.float64)
    weights = np.outer(sc.area, sc.area) * dense
    near = sphere @ sphere[ends].T > np.cos(np.radians(30.0))  # (2562, 4)
    masses = []
    for first, second in ((0, 2), (1, 3)):
        block = np.outer(near[:, first], near[:, second])
        masses.append(2 * weights[:2562, :2562][block].sum())  # and its mirror across the diagonal
    assert sum(masses) == pytest.approx(1.0, abs=0.02), "nearly all the mass lies near the ends"
    assert masses[0] == pytest.approx(masses[1], rel=0.02), masses


def test_a_basis_that_makes_a_signed_kernel_draws_a_warning(smoothable):
    """The toolkit's basis made a kernel 38% negative without a word (A2-02)."""
    modes = np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]]) / np.sqrt([[3], [2], [6]])
    signed = ((np.array([0.0, 0.1, 10.0]), modes.T),) * 2  # K[0, 1] = 1/3 - 0.95/2 < 0
    with pytest.warns(UserWarning, match="of the kernel's weight is negative"):
        smoothable.smooth(kernel="rdk", bandwidth=1.0, eigenpairs=signed)


@pytest.mark.parametrize(("nu", "kappa"), [(15.0, 0.6252), (200.0, 50.0), (100.0, 1000.0)])
def test_extreme_matern_settings_still_make_a_finite_unit_mass_density(
    smoothable, toy_basis, nu, kappa
):
    """An absolute floor zeroed every weight at nu 15; nu 200 at kappa 50 came back NaN (A2-05)."""
    out = smoothable.smooth(kernel="matern", bandwidth=kappa, nu=nu, eigenpairs=toy_basis)
    dense = out.dense(np.float64)
    assert np.isfinite(dense).all()
    assert smoothable.area @ dense @ smoothable.area == pytest.approx(1.0, rel=1e-6)


def test_a_kernel_that_is_zero_everywhere_is_refused_by_its_settings(smoothable):
    """The refusal said "check the endpoints" when the kernel's settings had zeroed it (A2-05)."""
    vectors = np.eye(3) / np.sqrt(2.0)
    no_zero_mode = ((np.array([0.5, 1.0, 1.5]), vectors),) * 2
    with pytest.raises(ValueError) as caught:
        smoothable.smooth(kernel="rdk", bandwidth=1000.0, eigenpairs=no_zero_mode)
    message = str(caught.value)
    assert "kernel='rdk', bandwidth=1000" in message and "zero everywhere" in message
    assert "not the endpoints" in message


def test_a_density_that_is_not_finite_is_refused_before_anything_can_save_it():
    """At nu 200 and kappa 50 smooth() returned all NaN, which save() wrote (A2-05)."""
    from sbci.smoothing import _finish

    class Fake:
        def __init__(self, data, area, mask, metadata, coords, endpoints):
            self.data, self.area, self.mask = data, area, mask
            self.metadata, self.coords, self.endpoints = metadata, coords, endpoints

    connectome = Fake(None, np.ones(4), np.ones(4, bool), metadata_template("sc"), None, None)
    dense = np.full((4, 4), np.nan)
    with pytest.raises(ValueError, match="kernel='matern', bandwidth=50, nu=200.* not finite"):
        _finish(connectome, dense, "matern", 50.0, nu=200.0)


def test_a_resmoothed_file_says_how_it_was_smoothed(smoothable, toy_basis, tmp_path):
    """The released provenance, "smoothed by sbci (kernel shk ...)", survived an rdk one (A2-09)."""
    import hashlib

    import scipy.io

    from sbci import smoothing

    smoothable.metadata.fields["provenance"] = (
        "HCP Young Adult sub-synth01; smoothed by sbci (kernel shk, bandwidth 0.005); "
        "endpoints from snapped_fibers.npz on the registered sphere; built"
    )
    out = smoothable.smooth(kernel="rdk", bandwidth=1.0, eigenpairs=toy_basis)
    said = out.metadata["provenance"]
    assert "kernel shk" not in said and "(kernel rdk, bandwidth 1, basis eigenpairs passed" in said
    assert said.startswith("HCP Young Adult sub-synth01; re-smoothed by sbci ")
    assert said.endswith("; endpoints from snapped_fibers.npz on the registered sphere; built")
    again = out.smooth(kernel="matern", bandwidth=2.0, nu=1.0, eigenpairs=toy_basis)
    assert again.metadata["provenance"].count("smoothed by sbci") == 1
    assert "(kernel matern, bandwidth 2, nu 1, basis" in again.metadata["provenance"]
    # read from files, the basis is named by each file's SHA-256 (A2-08)
    for (eigenvalues, vectors), side, name in zip(
        toy_basis, "LR", smoothing.EIGENPAIR_FILES, strict=True
    ):
        scipy.io.savemat(tmp_path / name, {f"Lambda_{side}": eigenvalues, f"U_{side}": vectors})
    filed = smoothable.smooth(kernel="rdk", bandwidth=1.0, eigenpairs=tmp_path)
    for name in smoothing.EIGENPAIR_FILES:
        digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert f"{name} sha256 {digest}" in filed.metadata["provenance"]


def test_a_sigma_below_the_width_floor_draws_a_warning():
    """Below sigma 0.001 the 33 terms, not sigma, set the width, without a word (A2-07)."""
    import warnings

    import sbci

    sc = sbci.example()
    sc.endpoints = sc.endpoints.take(np.arange(40))
    with pytest.warns(UserWarning, match="sigma 0.0005 is below 0.001"):
        sc.smooth(kernel="shk", bandwidth=0.0005)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sc.smooth(kernel="shk", bandwidth=0.001)

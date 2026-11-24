"""``sbci convert``: the pipeline's output into the package's files.

The inputs are made up, on the real grid, and laid out as SBCI_Pipeline_Py3 lays out a
run: ``<run>/sbci_output/files/work/<subject>/dwi_pipeline/sbci_connectome`` with the SC
as ``convert_raw.py`` writes it (``np.triu`` of a symmetric kernel, the diagonal's slot
kept, mass on the medial wall, ``c3_main``'s scale), the FC as ``calculate_fc.py`` writes
it (ones on the diagonal), ``snapped_fibers.npz`` with the registered spheres, the time
series with their runs, and ``<run>/sbci_output/files/grid/mapping_avg_ico4.npz``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io

import sbci
from sbci import convert as conversion
from sbci.cli import main
from sbci.grid import to_dense
from sbci.validate import validate_file

N = 5124
HALF = N // 2
SUBJECT = "sub-synth01"
ROOT = Path(__file__).resolve().parents[1]
PROFILE = """\
# A made-up profile, written as the pipeline's dataset profiles are.
DATASET_NAME=SYNTHETIC
INPUT_ROUTE=raw
DTI_BVALS=(0 1000)
FMRI_TR=0.8
FMRI_SKIP_SECONDS=6.4
FMRI_POLYFIT=2
FMRI_STC=none
FMRI_MOTION_REGRESSORS=tsv
N_RUNS=4
BANDWIDTH=0.005
ROIS=("4" "8" "10")
RESOLUTION=ico4
"""
CONTAINER = "registry.example/sbci-full@sha256:" + "0" * 64
PIPELINE = "SBCI_Pipeline_Py3@0123456789ab"


# --- a pipeline-shaped run --------------------------------------------------------------


def _grid():
    from sbci.atlas import cortex_mask
    from sbci.io.cifti import vertex_areas
    from sbci.surface import load_surface

    sphere = np.array(load_surface("sphere").vertices, dtype=np.float64)
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    return sphere, np.asarray(vertex_areas()[1], np.float64), cortex_mask()


def _discs(sphere, rng, count, size):
    """Pairs of vertex discs, each within one hemisphere."""
    side = np.arange(N) // HALF
    for _ in range(count):
        a, b = rng.integers(0, N, 2)
        yield (
            np.argsort(-(sphere @ sphere[a]) + 2 * (side != side[a]))[:size],
            np.argsort(-(sphere @ sphere[b]) + 2 * (side != side[b]))[:size],
        )


def structural_mat(rng, sphere, mask, diagonal=0.0, lower=False):
    """The pipeline's SC matrix: np.triu of a kernel with mass on the wall, its own scale."""
    K = np.zeros((N, N))
    for disc_a, disc_b in _discs(sphere, rng, 60, 40):
        K[np.ix_(disc_a, disc_b)] += rng.integers(1, 6)
    K = K + K.T
    K[np.ix_(~mask, mask)] += 0.05
    K[np.ix_(mask, ~mask)] += 0.05
    np.fill_diagonal(K, diagonal)
    K *= 1e-3
    upper = np.triu(K)
    if lower:
        upper[N - 1, 0] = 1e-3
    return upper


def functional_mat(rng, sphere):
    """The pipeline's FC matrix: the upper triangle, ones on the diagonal."""
    F = np.zeros((N, N))
    for disc_a, disc_b in _discs(sphere, rng, 80, 60):
        F[np.ix_(disc_a, disc_b)] = rng.choice([-0.5, -0.25, 0.25, 0.5, 0.75])
    F = np.triu(F, 1)
    F = F + F.T
    np.fill_diagonal(F, 1.0)
    return np.triu(F)


def write_vtk(path, points, binary=False, kind="float"):
    """A legacy VTK POLYDATA with these points, ASCII or big-endian binary."""
    head = f"# vtk DataFile Version 3.0\nvtk output\n{'BINARY' if binary else 'ASCII'}\n"
    head += f"DATASET POLYDATA\nPOINTS {len(points)} {kind}\n"
    if binary:
        body = np.asarray(points, dtype=">f4" if kind == "float" else ">f8").tobytes()
        Path(path).write_bytes(head.encode() + body + b"\nPOLYGONS 1 4\n")
    else:
        rows = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in points)
        Path(path).write_text(head + rows + "\nPOLYGONS 1 4\n3 0 1 2\n")


def write_mapping(path, area):
    """``mapping_avg_ico4.npz`` as the pipeline writes it: each vertex's fsaverage vertices."""
    counts = np.rint(area).astype(int)
    mapping = np.empty(N, dtype=object)
    stops = np.cumsum(counts)
    for i, (stop, count) in enumerate(zip(stops, counts, strict=True)):
        mapping[i] = np.arange(stop - count, stop)
    np.savez(path, mapping=mapping, shape=np.array([N, HALF, HALF]))


def write_series(path, frames, rng):
    np.savez_compressed(
        path,
        lh_time_series=rng.standard_normal((40, frames)).astype(np.float32),
        rh_time_series=rng.standard_normal((40, frames)).astype(np.float32),
        subcortical_labels=np.asarray(["label_4"], dtype="S"),
        label_4=rng.standard_normal((5, frames)),
    )


def make_run(root: Path, seed: int = 0) -> Path:
    """A whole run folder, as run_subject.sbatch leaves it before the SBCI package's calls."""
    rng = np.random.default_rng(seed)
    sphere, area, mask = _grid()
    folder = root / "sbci_output/files/work" / SUBJECT / "dwi_pipeline/sbci_connectome"
    grid = root / "sbci_output/files/grid"
    for path in (folder, grid, root / "config", root / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    write_mapping(grid / "mapping_avg_ico4.npz", area)
    scipy.io.savemat(
        folder / "smoothed_sc_avg_0.005_ico4.mat",
        {"sc": structural_mat(rng, sphere, mask)},
        do_compression=True,
    )
    scipy.io.savemat(
        folder / "fc_avg_ico4.mat",
        {"fc": functional_mat(rng, sphere), "sub_sub_fc": np.eye(3)},
        do_compression=True,
    )
    for run in ("RUN001", "RUN002"):
        (folder / run).mkdir()
        write_series(folder / run / "fc_ts.npz", 60, rng)
    write_series(folder / "fc_ts.npz", 120, rng)
    native = 3000  # each hemisphere's native vertices, on its registered sphere
    for side in ("lh", "rh"):
        points = rng.standard_normal((native, 3))
        write_vtk(
            folder / f"{side}_sphere_reg_lps.vtk",
            100 * points / np.linalg.norm(points, axis=1)[:, None],
        )
    count = 4000
    np.savez_compressed(
        folder / "snapped_fibers.npz",
        v_ids0=rng.integers(0, native, count),
        v_ids1=rng.integers(0, native, count),
        surf_ids0=rng.choice([0.0, 1.0, 2.0], count, p=[0.48, 0.48, 0.04]),
        surf_ids1=rng.choice([0.0, 1.0, 2.0], count, p=[0.48, 0.48, 0.04]),
    )
    endpoints = sbci.example("sc", seed=1).endpoints
    scipy.io.savemat(
        folder / "mesh_intersections_ico4.mat",
        {
            "surf_in": endpoints.surf_in.astype(float),
            "surf_out": endpoints.surf_out.astype(float),
            "tri_in": endpoints.tri_in + 1.0,
            "tri_out": endpoints.tri_out + 1.0,
            "vtx_in": endpoints.vtx_in + 1.0,
            "vtx_out": endpoints.vtx_out + 1.0,
            "pt_in": endpoints.bary_in.astype(float),
            "pt_out": endpoints.bary_out.astype(float),
        },
    )
    (root / "config" / "profile.conf").write_text(PROFILE)
    versions = {"pipeline_version": PIPELINE, "container_version": CONTAINER, "stages": "all"}
    (root / "config" / "versions.json").write_text(json.dumps(versions))
    (root / "config" / "sbci_local.json").write_text(json.dumps({"mode": "preprocess"}))
    (root / "local_adapter_manifest.json").write_text(json.dumps({"status": "complete"}))
    (root / "logs" / "sbci.err").write_text("a made-up message from the SBCI stage\n")
    return root


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """One run, built once and only read: a test that changes it works on a copy."""
    return make_run(tmp_path_factory.mktemp("pipeline") / "run")


@pytest.fixture
def run(built, tmp_path) -> Path:
    """A copy of the run, the test's to change."""
    return Path(shutil.copytree(built, tmp_path / "run", symlinks=True))


def connectome_folder(run: Path) -> Path:
    return run / "sbci_output/files/work" / SUBJECT / "dwi_pipeline/sbci_connectome"


def convert_args(run: Path, *extra: str) -> list[str]:
    return [
        "convert",
        str(connectome_folder(run)),
        "--subject",
        SUBJECT,
        "--out",
        str(run / "derivatives"),
        "--pipeline-version",
        PIPELINE,
        "--container-version",
        CONTAINER,
        "--config",
        str(run / "config" / "profile.conf"),
        *extra,
    ]


def convert(run: Path, capsys, *extra: str) -> tuple[int, dict, str]:
    status = main(convert_args(run, *extra))
    out, err = capsys.readouterr()
    return status, json.loads(out), err


def expected(path: Path, variable: str, modality: str) -> np.ndarray:
    """What a file should hold, computed apart from the converter.

    The strict upper triangle, the medial wall zeroed, SC scaled to unit mass with the
    bundled areas.
    """
    _, area, mask = _grid()
    matrix = scipy.io.loadmat(path)[variable].astype(np.float64)
    dense = np.triu(matrix, 1) + np.triu(matrix, 1).T
    dense[~mask, :] = 0.0
    dense[:, ~mask] = 0.0
    if modality == "sc":
        dense /= area @ dense @ area
    return dense


# --- pieces -----------------------------------------------------------------------------


def test_the_profile_is_read_as_text_and_never_run():
    profile = conversion.Profile.parse(
        "# a comment\n"
        "\n"
        "export BANDWIDTH=0.005   # trailing\n"
        "RESOLUTION='ico4'\n"
        'NAME="with space"\n'
        'ROIS=("4" "8" 10)\n'
        "EMPTY=\n"
        "LATER=1\n"
        "LATER=$OTHER\n"
        "TICKS=`date`\n"
        "TWO=a b\n"
        "source other.conf\n"
    )
    assert profile.get("BANDWIDTH") == "0.005"
    assert profile.get("RESOLUTION") == "ico4"
    assert profile.get("NAME") == "with space"
    assert profile.get("ROIS") == ("4", "8", "10")
    assert profile.get("EMPTY") == ""
    assert profile.get("MISSING") is None
    assert sorted(profile.unread) == ["LATER", "TICKS", "TWO"]  # the last LATER is bash's
    assert profile.other == ("source other.conf",)
    with pytest.raises(conversion.ProfileError, match="LATER"):
        profile.get("LATER")
    assert profile.settings("FMRI_") == {}
    with pytest.raises(conversion.ProfileError, match="TICKS"):
        profile.settings("TI")  # one in a form not read is refused, not left out


def test_the_pipelines_own_profile_layout_is_read():
    # the layout of SBCI_Pipeline_Py3's datasets/*/profile.conf, comments and arrays included
    text = (
        "####\n# HCP-like\n####\n\nDATASET_NAME=X\nSHELL_POLICY=all\nDTI_BVALS=(0 1000)\n"
        "FMRI_TR=0.72\nFMRI_SKIP_SECONDS=6.4\nFMRI_STC=none\nBANDWIDTH=0.005\n"
        'ROIS=("4" "8" "10" "11" "12" "13" "17" "18" "26" "43" "47" "49" "50" "51" "52" '
        '"53" "54" "58" "16")\n\nRESOLUTION=ico4\n'
    )
    profile = conversion.Profile.parse(text)
    assert len(profile.get("ROIS")) == 19 and profile.get("DTI_BVALS") == ("0", "1000")
    assert profile.settings("FMRI_") == {
        "FMRI_SKIP_SECONDS": "6.4",
        "FMRI_STC": "none",
        "FMRI_TR": "0.72",
    }
    assert not profile.unread and not profile.other


@pytest.mark.parametrize(
    ("name", "parsed"),
    [
        ("smoothed_sc_avg_0.005_ico4.mat", (0.005, "ico4")),
        ("smoothed_sc_avg_0.01_ico4.mat", (0.01, "ico4")),
        ("smoothed_sc_avg_1e-05_ico5.mat", (1e-05, "ico5")),
        ("smoothed_sc_avg_0_ico4.mat", None),
        ("smoothed_sc_avg_abc_ico4.mat", None),
        ("smoothed_sc_avg_0.005_ico4.npz", None),
        ("sub_sc_avg_0.005_ico4.mat", None),
    ],
)
def test_the_sc_name_gives_the_bandwidth_and_grid(name, parsed):
    assert conversion.parse_sc_name(name) == parsed


def test_output_names_take_the_subject_as_files_carry_it():
    assert conversion.output_name("synth01", "sc") == "sub-synth01_sc.h5"
    assert conversion.output_name("sub-synth01", "sc") == "sub-synth01_sc.h5"
    assert conversion.output_name("SUB-synth01", "fc") == "sub-synth01_fc.h5"
    assert conversion.output_name("synth01", "fc", session="ses-2") == "sub-synth01_ses-2_fc.h5"
    assert conversion.output_name("synth01", "sc", session="2") == "sub-synth01_ses-2_sc.h5"
    for subject, session in (
        ("ses-01", None),
        ("a/b", None),
        ("", None),
        ("x", ""),
        ("x_ses-1", None),
    ):
        with pytest.raises(ValueError):
            conversion.output_name(subject, "sc", session=session)


def test_clean_drops_the_diagonal_and_wall_and_scales_sc_to_unit_mass():
    rng = np.random.default_rng(3)
    n = 7
    area = rng.uniform(1, 3, n)
    mask = np.array([True, True, False, True, True, False, True])
    full = rng.uniform(0, 1, (n, n))
    full = full + full.T
    matrix = np.triu(full, 1)
    out = conversion.clean(matrix.copy(), "sc", area, mask)
    truth = np.triu(full, 1) + np.triu(full, 1).T
    total = area @ truth @ area
    truth[~mask, :] = 0
    truth[:, ~mask] = 0
    assert out.medial_wall == pytest.approx(1 - (area @ truth @ area) / total)
    assert out.diagonal == 0.0
    truth /= area @ truth @ area
    np.testing.assert_allclose(to_dense(out.data, n), truth, rtol=1e-6)

    signed = np.triu(full - 2.0, 1) + np.eye(n)  # an FC: signed, ones on the diagonal
    out = conversion.clean(signed.copy(), "fc", area, mask)
    absolute = np.abs(signed + np.triu(signed, 1).T)
    assert out.diagonal == pytest.approx(float(area**2 @ np.ones(n)) / (area @ absolute @ area))
    np.testing.assert_allclose(out.data[0], signed[0, 1], rtol=1e-6)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m.__setitem__((3, 1), 1.0), "lower triangle"),
        (lambda m: m.__setitem__((1, 3), np.nan), "not finite"),
        (lambda m: m.__setitem__((2, 2), 5.0), "diagonal holds"),
        (lambda m: m.__imul__(0.0), "no connectivity"),
    ],
)
def test_clean_refuses_what_the_pipeline_never_writes(change, message):
    matrix = np.triu(np.ones((4, 4)), 1)
    change(matrix)
    with pytest.raises(conversion.ConversionError, match=message):
        conversion.clean(matrix, "sc", np.ones(4), np.ones(4, bool))


def test_a_tiny_diagonal_passes_and_is_reported():
    matrix = np.triu(np.ones((4, 4)), 1)
    matrix[0, 0] = 1e-8  # below the limit's share of the mass
    out = conversion.clean(matrix, "sc", np.ones(4), np.ones(4, bool))
    assert 0 < out.diagonal < conversion.DIAGONAL_LIMIT


def test_rows_condense_in_the_grids_own_order():
    from sbci.grid import to_condensed

    matrix = np.random.default_rng(0).random((9, 9))
    np.testing.assert_array_equal(
        conversion._condensed(matrix), to_condensed(matrix).astype(np.float32)
    )


@pytest.mark.parametrize(("binary", "kind"), [(False, "float"), (True, "float"), (True, "double")])
def test_vtk_points_are_read_ascii_or_binary(tmp_path, binary, kind):
    points = np.random.default_rng(1).standard_normal((11, 3)).round(4)
    write_vtk(tmp_path / "s.vtk", points, binary=binary, kind=kind)
    np.testing.assert_allclose(conversion.read_vtk_points(tmp_path / "s.vtk"), points, atol=1e-5)
    (tmp_path / "t.vtk").write_text("# vtk DataFile Version 3.0\nx\nASCII\nPOINTS 3 float\n1 2 3\n")
    with pytest.raises(conversion.ConversionError, match="end before"):
        conversion.read_vtk_points(tmp_path / "t.vtk")


def test_frames_come_from_the_series_headers(tmp_path):
    rng = np.random.default_rng(0)
    write_series(tmp_path / "fc_ts.npz", 37, rng)
    assert conversion.frames(tmp_path / "fc_ts.npz") == 37
    np.savez(tmp_path / "bad.npz", lh_time_series=np.zeros((3, 5)), rh_time_series=np.zeros((3, 6)))
    with pytest.raises(conversion.ConversionError, match="disagree"):
        conversion.frames(tmp_path / "bad.npz")


# --- the command ------------------------------------------------------------------------


def test_convert_writes_both_files_and_they_pass_validate(run, capsys):
    status, report, err = convert(run, capsys)
    assert status == 0, report["errors"]
    assert report["status"] == "written" and err == ""
    assert set(report["written"]) == {"sc", "fc"} and report["skipped"] == {}
    folder = connectome_folder(run)
    for modality, variable, name in (
        ("sc", "sc", "smoothed_sc_avg_0.005_ico4.mat"),
        ("fc", "fc", "fc_avg_ico4.mat"),
    ):
        path = run / "derivatives" / f"{SUBJECT}_{modality}.h5"
        assert report["written"][modality] == str(path.resolve())
        assert all(check.passed for check in validate_file(path))
        assert main(["validate", str(path)]) == 0
        cc = sbci.load(path)
        np.testing.assert_allclose(
            cc.dense(np.float64), expected(folder / name, variable, modality), rtol=2e-7, atol=0
        )
        fields = cc.metadata.fields
        assert fields["pipeline_version"] == PIPELINE
        assert fields["container_version"] == CONTAINER
        assert fields["entry_point"] == "precomputed"
        assert fields["conversion"] == report["removed"][modality]
        digests = {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in (name,)}
        assert fields["inputs"][name] == digests[name]
    capsys.readouterr()

    sc = sbci.load(run / "derivatives" / f"{SUBJECT}_sc.h5")
    assert sc.metadata["bandwidth"] == 0.005 and sc.metadata["kernel"] == "shk"
    assert sc.metadata["streamline_weighting"] == "none"
    snapped = np.load(folder / "snapped_fibers.npz")
    on_white = (snapped["surf_ids0"] <= 1) & (snapped["surf_ids1"] <= 1)
    assert sc.endpoints.n_streamlines == sc.metadata["streamline_count"] == int(on_white.sum())
    assert sc.metadata["endpoint_source"].startswith("snapped_fibers.npz")
    assert report["removed"]["sc"]["diagonal_share"] == 0.0
    assert 0 < report["removed"]["sc"]["medial_wall_share"] < 1

    fc = sbci.load(run / "derivatives" / f"{SUBJECT}_fc.h5")
    assert fc.metadata["fc_frames"] == 120
    assert fc.metadata["fc_runs"] == ["RUN001", "RUN002"]
    assert "FMRI_POLYFIT=2" in fc.metadata["fc_nuisance_model"]
    assert fc.metadata["normalization"] == "none"
    assert report["removed"]["fc"]["diagonal_share"] > 0  # the ones, dropped
    assert {i["file"] for i in report["inputs"]} >= {
        "profile.conf",
        "mapping_avg_ico4.npz",
        "snapped_fibers.npz",
        "fc_ts.npz",
    }
    assert not list((run / "derivatives").glob(".*")), "a temporary file was left behind"


def test_a_modality_without_inputs_is_skipped(run, capsys):
    folder = connectome_folder(run)
    (folder / "fc_avg_ico4.mat").unlink()
    status, report, _ = convert(run, capsys)
    assert status == 0 and set(report["written"]) == {"sc"}
    assert "fc_avg" in report["skipped"]["fc"]
    assert not (run / "derivatives" / f"{SUBJECT}_fc.h5").exists()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("lower", "lower triangle"),
        ("diagonal", "diagonal holds"),
        ("bandwidth", "BANDWIDTH=0.01"),
        ("resolution", "RESOLUTION=ico5"),
        ("mapping", "disagrees with the bundled vertex areas"),
    ],
)
def test_an_unusable_input_exits_1_and_writes_no_sc(run, capsys, case, message):
    folder = connectome_folder(run)
    sphere, area, mask = _grid()
    rng = np.random.default_rng(5)
    if case in ("lower", "diagonal"):
        matrix = structural_mat(
            rng, sphere, mask, diagonal=0.5 * (case == "diagonal"), lower=case == "lower"
        )
        scipy.io.savemat(folder / "smoothed_sc_avg_0.005_ico4.mat", {"sc": matrix})
    elif case in ("bandwidth", "resolution"):
        text = PROFILE.replace("BANDWIDTH=0.005", "BANDWIDTH=0.01")
        if case == "resolution":
            text = PROFILE.replace("RESOLUTION=ico4", "RESOLUTION=ico5")
        (run / "config" / "profile.conf").write_text(text)
    else:
        moved = area.copy()
        moved[0] += 1
        moved[1] -= 1
        write_mapping(run / "sbci_output/files/grid/mapping_avg_ico4.npz", moved)
    status, report, err = convert(run, capsys)
    assert status == 1 and report["status"] == "failed"
    assert any(message in error for error in report["errors"]), report["errors"]
    assert message in err
    assert not (run / "derivatives" / f"{SUBJECT}_sc.h5").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda run: ["--subject", "ses-01"], "session label"),
        (lambda run: ["--config", str(run / "nowhere.conf")], "is not a file"),
        (lambda run: ["--mapping", str(run / "nowhere.npz")], "is not a file"),
    ],
)
def test_a_usage_error_exits_2(built, capsys, change, message):
    status = main(convert_args(built, *change(built)))
    out, err = capsys.readouterr()
    assert status == 2 and message in err
    assert json.loads(out)["status"] == "usage"
    assert not (built / "derivatives").exists()


def test_argparse_usage_errors_exit_2(built, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["convert", str(connectome_folder(built)), "--out", str(tmp_path)])  # no --subject
    assert stopped.value.code == 2
    status = main(["convert", str(tmp_path / "missing"), "--subject", "x", "--out", str(tmp_path)])
    assert status == 2 and "is not a folder" in capsys.readouterr().err


def test_a_folder_with_nothing_to_convert_exits_1(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    status = main(
        ["convert", str(tmp_path / "empty"), "--subject", "x", "--out", str(tmp_path / "d")]
    )
    report = json.loads(capsys.readouterr().out)
    assert status == 1 and "nothing to convert" in " ".join(report["errors"])


def test_unsnapped_endpoints_are_used_and_said_so_without_the_snapped_ones(run, capsys):
    (connectome_folder(run) / "snapped_fibers.npz").unlink()
    status, report, err = convert(
        run, capsys, "--container-version", "sbci-full:1.0", "--session", "2"
    )
    assert status == 0
    sc = sbci.load(run / "derivatives" / f"{SUBJECT}_ses-2_sc.h5")
    assert "unsnapped" in sc.metadata["endpoint_source"]
    assert sc.endpoints.n_streamlines == sbci.example("sc", seed=1).endpoints.n_streamlines
    assert any("unsnapped" in w for w in report["warnings"])
    assert any("@sha256:" in w for w in report["warnings"]) and "warning:" in err
    assert sc.metadata["container_version"] == "sbci-full:1.0"


def test_a_file_that_fails_validate_is_not_kept(run, capsys, monkeypatch):
    from sbci import validate

    def failing(path, tolerance=1e-5):
        return [validate.Check("readable", True), validate.Check("unit mass", False, "made up")]

    monkeypatch.setattr(validate, "validate_file", failing)
    status, report, _ = convert(run, capsys)
    assert status == 1 and report["written"] == {}
    assert any("not kept" in e and "made up" in e for e in report["errors"])
    assert not (run / "derivatives" / f"{SUBJECT}_sc.h5").exists()
    assert not [p for p in (run / "derivatives").iterdir()], "a temporary file was left behind"


# --- the pipeline's own command lines ---------------------------------------------------

# SBCI_Pipeline_Py3, workflow/run_subject.sbatch, lines 188-197 at eb04228, byte for byte
# (PIPELINE_LINES_SHA256; the pieces only keep the lines short here). python3 is the
# interpreter under test, a shell function, and the job's variables are in the environment.
# set -u is left out: bash before 4.4, macOS's, applies it to an empty array.
PIPELINE_LINES = "".join(
    [
        "    export MPLBACKEND=Agg\n",
        "    python3 -m sbci convert"
        ' "$OUT/sbci_output/files/work/$SUBJ/dwi_pipeline/sbci_connectome" --subject "$SUBJ" \\\n',
        '        --out "$OUT/derivatives" --pipeline-version "$PIPELINE_VERSION"'
        ' --container-version "$CONTAINER_VERSION" \\\n',
        '        ${PROFILE:+--config "$PROFILE"} > "$OUT/logs/convert.json"'
        ' 2> "$OUT/logs/convert.err" || rc=$?\n',
        "    H5=()\n",
        '    [ -s "$OUT/derivatives/${SUBJ}_sc.h5" ]'
        ' && H5+=(--sc "$OUT/derivatives/${SUBJ}_sc.h5")\n',
        '    [ -s "$OUT/derivatives/${SUBJ}_fc.h5" ]'
        ' && H5+=(--fc "$OUT/derivatives/${SUBJ}_fc.h5")\n',
        '    python3 -m sbci qc "${H5[@]}" --run-dir "$OUT" --out "$OUT/qc" > "$OUT/logs/qc.json"'
        ' 2> "$OUT/logs/qc.err" \\\n',
        '        || echo "QC flagged a check: see $OUT/qc/index.html"\n',
        '    ls "$OUT/derivatives" "$OUT/qc/index.html" 2>/dev/null\n',
    ]
)
PIPELINE_LINES_SHA256 = "cf1855f6106637dd798835cfc47b101f41fc590b89785063159c7ab1c27a1528"


def test_the_pipeline_lines_are_the_pipelines():
    assert hashlib.sha256(PIPELINE_LINES.encode()).hexdigest() == PIPELINE_LINES_SHA256


def run_pipeline_lines(run: Path) -> subprocess.CompletedProcess:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash to run the pipeline's lines")
    script = (
        'set -o pipefail\npython3() { "$SBCI_TEST_PYTHON" "$@"; }\nrc=0\n'
        + PIPELINE_LINES
        + 'echo "rc=$rc"\nexit "$rc"\n'
    )
    environment = dict(
        os.environ,
        OUT=str(run),
        SUBJ=SUBJECT,
        PIPELINE_VERSION=PIPELINE,
        CONTAINER_VERSION=CONTAINER,
        PROFILE=str(run / "config" / "profile.conf"),
        SBCI_TEST_PYTHON=sys.executable,
    )
    return subprocess.run(
        [bash, "-c", script], env=environment, capture_output=True, text=True, timeout=600
    )


def test_the_pipelines_lines_write_both_files_and_the_qc_page(run):
    done = run_pipeline_lines(run)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "QC flagged" not in done.stdout and "rc=0" in done.stdout
    convert_report = json.loads((run / "logs" / "convert.json").read_text())
    assert convert_report["status"] == "written"
    for modality in ("sc", "fc"):
        assert all(
            c.passed for c in validate_file(run / "derivatives" / f"{SUBJECT}_{modality}.h5")
        )
    verdict = json.loads((run / "logs" / "qc.json").read_text())
    assert verdict["status"] == "pass", [c for c in verdict["checks"] if not c["passed"]]
    assert verdict["limits"] == "no reference limits installed"
    page = (run / "qc" / "index.html").read_text()
    assert "PASS" in page and "a made-up message from the SBCI stage" in page
    assert verdict["versions"]["run"]["pipeline_version"] == PIPELINE


def test_the_pipelines_lines_when_the_sc_is_refused(run):
    sphere, _, mask = _grid()
    matrix = structural_mat(np.random.default_rng(2), sphere, mask, lower=True)
    scipy.io.savemat(connectome_folder(run) / "smoothed_sc_avg_0.005_ico4.mat", {"sc": matrix})
    done = run_pipeline_lines(run)
    assert done.returncode == 1 and "rc=1" in done.stdout
    assert "QC flagged a check" in done.stdout
    assert not (run / "derivatives" / f"{SUBJECT}_sc.h5").exists()
    assert (run / "derivatives" / f"{SUBJECT}_fc.h5").exists()
    verdict = json.loads((run / "logs" / "qc.json").read_text())
    convert_check = next(c for c in verdict["checks"] if c["name"] == "convert")
    assert verdict["status"] == "fail" and "lower triangle" in convert_check["detail"]
    assert (run / "qc" / "index.html").exists()


def test_the_pipelines_lines_with_no_connectivity_still_write_the_page(run):
    shutil.rmtree(connectome_folder(run))
    connectome_folder(run).mkdir()
    done = run_pipeline_lines(run)
    assert done.returncode == 1 and "QC flagged a check" in done.stdout
    verdict = json.loads((run / "logs" / "qc.json").read_text())
    assert verdict["status"] == "fail"
    assert "no file was given" in (run / "qc" / "index.html").read_text()

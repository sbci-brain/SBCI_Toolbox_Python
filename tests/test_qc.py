"""``sbci qc``: one subject's page and verdict, and the reference limits it reads."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import sbci
from sbci import qc as quality
from sbci.cli import main
from sbci.io import write_hdf5

SUBJECT = "sub-synth02"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> tuple[Path, Path]:
    """A made-up subject's two files, on the real grid."""
    folder = tmp_path_factory.mktemp("qc")
    sc = sbci.example("sc", seed=3).save(folder / f"{SUBJECT}_sc.h5")
    fc = sbci.example("fc", seed=4).save(folder / f"{SUBJECT}_fc.h5")
    return Path(sc), Path(fc)


def verdict(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_the_measures_are_what_they_say(files):
    sc, fc = (sbci.load(path) for path in files)
    values = quality.measures(sc, fc)
    assert set(values) == set(quality.MEASURES)
    dense = sc.dense(np.float64)
    area = sc.area
    total = area @ dense @ area
    half = sc.n_vertices // 2
    between = (
        area[:half] @ dense[:half, half:] @ area[half:]
        + area[half:] @ dense[half:, :half] @ area[:half]
    )
    assert values["sc_interhemispheric_share"] == pytest.approx(between / total)
    assert values["sc_max_vertex_share"] == pytest.approx(np.max(area * (dense @ area)) / total)
    assert values["sc_streamline_count"] == sc.metadata["streamline_count"]
    del dense
    cortex = np.asarray(fc.mask, bool)
    block = fc.dense(np.float64)[np.ix_(cortex, cortex)]
    pairs = block[np.triu_indices(int(cortex.sum()), 1)]
    assert values["fc_mean"] == pytest.approx(pairs.mean())
    assert values["fc_sd"] == pytest.approx(pairs.std())
    assert values["fc_frames"] is None  # the example records none
    coupling = sbci.structure_function_coupling(sc, fc)
    on = cortex & np.isfinite(coupling)
    assert values["coupling_median"] == pytest.approx(np.median(coupling[on]))


def test_good_files_pass_and_the_page_says_no_limits_are_installed(files, tmp_path, capsys):
    sc, fc = files
    status = main(["qc", "--sc", str(sc), "--fc", str(fc), "--out", str(tmp_path / "qc")])
    result = verdict(capsys)
    assert status == 0 and result["status"] == "pass"
    assert result["limits"] == "no reference limits installed"
    assert result["subject"] == SUBJECT
    page = Path(result["page"]).read_text()
    assert "PASS: every check passed" in page and "no reference limits installed" in page
    for name in ("sc: validate unit mass", "fc: validate correlation", "same subject", "same grid"):
        assert name in page and any(c["name"] == name and c["passed"] for c in result["checks"])
    for measure in quality.MEASURES:
        assert measure in page
    assert "<h2>Files</h2>" in page and "synthetic example" in page  # each file's own record
    try:
        import matplotlib  # noqa: F401
        import nilearn  # noqa: F401
    except ImportError:
        pytest.skip("the plotting extra is not installed: the figures are not drawn")
    assert (
        page.count("data:image/png;base64,") == 6
    )  # three surface maps, two matrices, a histogram


def test_no_file_still_writes_the_page_and_exits_1(tmp_path, capsys):
    status = main(["qc", "--out", str(tmp_path / "qc")])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert status == 1 and result["status"] == "fail"
    assert "no file was given" in (tmp_path / "qc" / "index.html").read_text()
    assert "sbci qc: fail" in captured.err


def test_without_matplotlib_the_page_has_no_figures(files, tmp_path, capsys, monkeypatch):
    for name in ("matplotlib", "matplotlib.figure", "matplotlib.pyplot", "nilearn"):
        monkeypatch.setitem(sys.modules, name, None)  # an import of it fails
    status = main(["qc", "--sc", str(files[0]), "--out", str(tmp_path / "qc")])
    assert status == 0 and verdict(capsys)["status"] == "pass"
    page = (tmp_path / "qc" / "index.html").read_text()
    assert "<img" not in page and "matplotlib is not installed" in page


def test_a_file_that_fails_validate_fails(files, tmp_path):
    cc = sbci.load(files[0])
    doubled = tmp_path / f"{SUBJECT}_sc.h5"
    write_hdf5(doubled, data=cc.data * 2, area=cc.area, mask=cc.mask, metadata=cc.metadata)
    result = quality.qc(sc=doubled, out=tmp_path / "qc", figures=False)
    failed = [c["name"] for c in result.checks if not c["passed"]]
    assert result.status == "fail" and failed == ["sc: validate unit mass"]
    assert "FAIL: a check failed" in result.page.read_text()


def test_files_that_do_not_belong_together_fail(files, tmp_path):
    sc, fc = files
    other = shutil.copy(fc, tmp_path / "sub-synth03_fc.h5")
    result = quality.qc(sc=sc, fc=other, out=tmp_path / "a", figures=False)
    assert [c["name"] for c in result.checks if not c["passed"]] == ["same subject"]
    result = quality.qc(sc=fc, out=tmp_path / "b", figures=False)  # an FC given as the SC
    assert [c["name"] for c in result.checks if not c["passed"]] == ["sc: modality"]
    assert result.status == "fail"


def run_folder(root: Path) -> Path:
    """The records a pipeline run leaves, made up."""
    for folder in ("config", "logs", "sbci_output"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    versions = {
        "pipeline_version": "SBCI_Pipeline_Py3@0123456789ab",
        "container_version": "x@sha256:0",
    }
    (root / "config" / "versions.json").write_text(json.dumps(versions))
    (root / "config" / "sbci_local.json").write_text("{not json")
    manifest = {"status": "complete", "issues": [{"code": "made_up", "message": "an issue"}]}
    (root / "sbci_output" / "local_adapter_manifest.json").write_text(json.dumps(manifest))
    (root / "logs" / "sbci.err").write_text("first line\nlast line of the stage's errors\n")
    (root / "logs" / "empty.err").write_text("")
    (root / "logs" / "qc.err").write_text("this command's own stderr\n")
    report = {"status": "written", "written": {"sc": "x"}, "errors": []}
    (root / "logs" / "convert.json").write_text(json.dumps(report))
    return root


def test_the_runs_records_are_shown(files, tmp_path):
    run = run_folder(tmp_path / "run")
    result = quality.qc(sc=files[0], run_dir=run, out=run / "qc", figures=False)
    page = result.page.read_text()
    assert result.status == "pass", [c for c in result.checks if not c["passed"]]
    assert "last line of the stage&#x27;s errors" in page
    assert "this command&#x27;s own stderr" not in page and "logs/empty.err" not in page
    assert "SBCI_Pipeline_Py3@0123456789ab" in page and "made_up: an issue" in page
    assert 'config/sbci_local.json</code></h3><p class="fail">' in page  # not JSON, said so
    versions = result.as_dict()["versions"]
    assert versions["run"]["pipeline_version"] == "SBCI_Pipeline_Py3@0123456789ab"
    assert versions["files"]["sc"]["container_version"].startswith("none (synthetic")


@pytest.mark.parametrize(
    ("record", "detail"),
    [
        ({"status": "failed", "errors": ["sc: connectivity: made up"]}, "made up"),
        ("", "stopped before reporting"),
    ],
)
def test_a_conversion_that_failed_fails(files, tmp_path, record, detail):
    run = run_folder(tmp_path / "run")
    text = record if isinstance(record, str) else json.dumps(record)
    (run / "logs" / "convert.json").write_text(text)
    result = quality.qc(sc=files[0], run_dir=run, out=run / "qc", figures=False)
    convert = next(c for c in result.checks if c["name"] == "convert")
    assert result.status == "fail" and not convert["passed"] and detail in convert["detail"]


def test_the_page_is_written_whatever_stops_the_checks(files, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("made up")

    monkeypatch.setattr(quality, "survey", broken)
    result = quality.qc(sc=files[0], out=tmp_path / "a", figures=False)
    assert result.status == "fail"
    assert "the checks stopped: RuntimeError: made up" in result.page.read_text()
    monkeypatch.setattr(quality, "render", broken)
    result = quality.qc(sc=files[0], out=tmp_path / "b", figures=False)
    assert "could not be drawn: RuntimeError: made up" in result.page.read_text()


def limits_file(path: Path, table: dict, version: int = quality.MEASURES_VERSION) -> Path:
    document = {
        "format": quality.LIMITS_FORMAT,
        "measures_version": version,
        "quantiles": [0.01, 0.99],
        "provenance": {"source": "made up for a test", "n_subjects": 3},
        "limits": table,
    }
    path.write_text(json.dumps(document))
    return path


def test_limits_flag_a_measure_outside_them(files, tmp_path, monkeypatch, capsys):
    value = quality.measures(sc=sbci.load(files[0]))["sc_interhemispheric_share"]
    inside = limits_file(tmp_path / "in.json", {"sc_interhemispheric_share": {"low": 0, "high": 1}})
    outside = limits_file(
        tmp_path / "out.json", {"sc_interhemispheric_share": {"low": value + 0.01, "high": 1}}
    )
    monkeypatch.setenv(quality.LIMITS_ENVIRONMENT, str(inside))
    result = quality.qc(sc=files[0], out=tmp_path / "a", figures=False)
    assert result.status == "pass" and result.limits_from == str(inside)
    assert "made up for a test, 3 subjects" in result.page.read_text()
    status = main(
        ["qc", "--sc", str(files[0]), "--out", str(tmp_path / "b"), "--limits", str(outside)]
    )
    flagged = verdict(capsys)
    row = next(m for m in flagged["measures"] if m["name"] == "sc_interhemispheric_share")
    assert status == 1 and flagged["status"] == "flag" and row["within"] is False
    assert "OUTSIDE" in (tmp_path / "b" / "index.html").read_text()

    stale = limits_file(tmp_path / "old.json", {}, version=quality.MEASURES_VERSION + 1)
    monkeypatch.setenv(quality.LIMITS_ENVIRONMENT, str(stale))
    result = quality.qc(sc=files[0], out=tmp_path / "c", figures=False)
    assert result.status == "fail" and "rebuild it" in result.limits_from
    monkeypatch.setenv(quality.LIMITS_ENVIRONMENT, str(tmp_path / "missing.json"))
    assert quality.qc(sc=files[0], out=tmp_path / "d", figures=False).status == "fail"


def test_usage_errors_exit_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["qc", "--sc", "x.h5"])  # no --out
    assert stopped.value.code == 2
    assert main(["qc", "--out", str(tmp_path), "--limits", str(tmp_path / "none.json")]) == 2
    assert "is not a file" in capsys.readouterr().err


def test_a_page_that_cannot_be_written_exits_1(tmp_path, capsys):
    (tmp_path / "taken").write_text("a file, not a folder")
    assert main(["qc", "--out", str(tmp_path / "taken")]) == 1
    assert json.loads(capsys.readouterr().out)["page"] is None


def test_the_reference_tool_writes_limits_qc_reads(files, tmp_path, capsys, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import build_qc_reference
    finally:
        sys.path.remove(str(ROOT / "tools"))
    population = tmp_path / "population"
    population.mkdir()
    for k in range(3):  # three made-up subjects: the mechanics, not limits anyone should use
        for path in files:
            shutil.copy(path, population / path.name.replace(SUBJECT, f"sub-synthr{k}"))
    out = tmp_path / "limits.json"
    common = [str(population), "--source", "made-up files", "--out", str(out)]
    assert build_qc_reference.main([*common, "--expect", "4"]) == 1 and not out.exists()
    assert "not the 4 expected" in capsys.readouterr().err

    # Each subject's measures as qc takes them, made to differ so the quantiles show.
    real = quality.measures(*(sbci.load(path) for path in files))
    calls = []

    def measured(sc, fc):
        assert sc.modality == "sc" and fc.modality == "fc"  # the two files of one subject
        calls.append(1)
        return {name: None if value is None else value * len(calls) for name, value in real.items()}

    monkeypatch.setattr(quality, "measures", measured)
    assert build_qc_reference.main([*common, "--expect", "3"]) == 0 and len(calls) == 3
    monkeypatch.undo()
    document = json.loads(out.read_text())
    assert document["format"] == quality.LIMITS_FORMAT
    assert document["measures_version"] == quality.MEASURES_VERSION
    assert document["provenance"]["n_subjects"] == 3 and document["quantiles"] == [0.01, 0.99]
    assert "sub-synthr0" not in out.read_text()  # the digest identifies the files, not the ids
    assert set(document["limits"]) == {name for name, value in real.items() if value is not None}
    for name, row in document["limits"].items():
        column = [real[name] * k for k in (1, 2, 3)]
        assert row["n"] == 3 and row["min"] == pytest.approx(min(column))
        assert row["low"] == pytest.approx(np.quantile(column, 0.01))
        assert row["high"] == pytest.approx(np.quantile(column, 0.99))
    result = quality.qc(sc=files[0], fc=files[1], out=tmp_path / "qc", limits=out, figures=False)
    assert result.limits["provenance"]["source"] == "made-up files"
    measured_rows = {m["name"]: m for m in result.measures if m["value"] is not None}
    assert set(measured_rows) == set(document["limits"])
    # the real subject's value is the end of the three nearest zero, so outside the 1% to 99%
    # range wherever it is not zero, and flagged there
    flagged = {n for n, m in measured_rows.items() if m["within"] is False}
    assert flagged == {n for n in measured_rows if real[n] != 0}
    assert result.status == ("flag" if flagged else "pass")

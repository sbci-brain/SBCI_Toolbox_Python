"""The metadata contract: a file with missing keys must not load."""

from __future__ import annotations

import numpy as np
import pytest

from sbci.metadata import Metadata, MetadataError, template


def test_template_sc_is_incomplete_until_filled():
    """A template left unfilled fails, rather than defaulting to plausible values."""
    with pytest.raises(MetadataError, match="missing required metadata keys"):
        template("sc").validate()


def test_complete_metadata_validates(sc_metadata):
    sc_metadata.validate()


def test_fc_does_not_require_streamline_keys():
    metadata = template(
        "fc",
        normalization="none",
        registration_reference="fsaverage",
        pipeline_version="0.0.1.dev0",
        container_version="sbci.sif@sha256:0",
        fc_nuisance_model="24P+aCompCor",
    )
    metadata.validate()
    assert "streamline_count" not in metadata.missing_keys()


def test_sc_does_not_require_the_fc_nuisance_model(sc_metadata):
    assert "fc_nuisance_model" not in sc_metadata.missing_keys()


def test_retired_grid_is_rejected(sc_metadata):
    sc_metadata.fields["grid"] = "fsaverage-ico4_0.94"
    with pytest.raises(MetadataError, match="retired"):
        sc_metadata.validate()


def test_hemisphere_order_is_enforced(sc_metadata):
    sc_metadata.fields["hemisphere_order"] = ["R", "L"]
    with pytest.raises(MetadataError, match="hemisphere_order"):
        sc_metadata.validate()


def test_unknown_modality_is_rejected(sc_metadata):
    sc_metadata.fields["included_connections"] = "ec"
    with pytest.raises(MetadataError, match="included_connections"):
        sc_metadata.validate()


def test_json_round_trip(sc_metadata):
    restored = Metadata.from_json(sc_metadata.to_json())
    restored.validate()
    assert restored.fields == sc_metadata.fields


def test_from_json_rejects_a_bare_array():
    with pytest.raises(MetadataError, match="JSON object"):
        Metadata.from_json("[1, 2, 3]")


def test_template_rejects_unknown_modality():
    with pytest.raises(ValueError, match="modality must be one of"):
        template("dti")


def test_the_storage_convention_and_spec_version_are_checked():
    from sbci.metadata import MetadataError, template

    good = template(
        "fc",
        normalization="none",
        registration_reference="fsaverage",
        pipeline_version="x",
        container_version="y",
        fc_nuisance_model="36p",
    )
    good.validate()
    bad = template("fc", **{**good.fields, "storage_convention": "lower-triangular-float64"})
    with pytest.raises(MetadataError, match="storage_convention"):
        bad.validate()
    old = template("fc", **{**good.fields, "spec_version": "9.0.0"})
    with pytest.raises(MetadataError, match="spec_version"):
        old.validate()


def test_json_round_trip_handles_numpy_scalars_and_refuses_nan():
    from sbci.metadata import Metadata, MetadataError

    text = Metadata({"bandwidth": np.float32(0.005), "count": np.int64(3)}).to_json()
    back = Metadata.from_json(text)
    assert back["count"] == 3 and abs(back["bandwidth"] - 0.005) < 1e-6
    with pytest.raises(MetadataError, match="serialized"):
        Metadata({"bandwidth": float("nan")}).to_json()
    with pytest.raises(MetadataError, match="JSON"):
        Metadata.from_json("{not json")


def test_a_file_without_a_spec_version_is_a_metadata_error_not_a_key_error():
    fields = dict(
        template(
            "fc",
            normalization="none",
            registration_reference="fsaverage",
            pipeline_version="test",
            container_version="test",
            fc_nuisance_model="none",
        ).fields
    )
    del fields["spec_version"]
    with pytest.raises(MetadataError, match="no spec_version"):
        Metadata(fields).validate()


def test_nu_belongs_to_a_matern_file_and_is_a_positive_number(sc_metadata):
    """smooth() records the Matern smoothness, and load_cohort sets files apart by it."""
    template("sc", **{**sc_metadata.fields, "kernel": "matern", "nu": 2.0}).validate()
    for changes, message in (
        ({"kernel": "rdk", "nu": 2.0}, "but this file's kernel is 'rdk'"),
        ({"kernel": "matern", "nu": 0}, "positive number, got 0"),
        ({"kernel": "matern", "nu": "3"}, "positive number, got '3'"),
        ({"kernel": "matern", "nu": True}, "positive number, got True"),
    ):
        with pytest.raises(MetadataError, match=message):
            template("sc", **{**sc_metadata.fields, **changes}).validate()


# --- the review of 23 November 2026 (A1-01, A1-04, A1-15) -------------------------------


def _fc(**changes):
    fields = dict(
        normalization="none",
        registration_reference="fsaverage",
        pipeline_version="p",
        container_version="c",
        fc_nuisance_model="none",
    )
    return template("fc", **{**fields, **changes})


@pytest.mark.parametrize(
    ("version", "family"),
    [
        ("0.1.0-draft", "0.1"),
        ("0.1.0", "0.1"),
        ("0.1.7", "0.1"),
        ("1.0.0", "1.0"),
        ("10.2.3", "10.2"),
    ],
)
def test_a_version_names_its_format(version, family):
    from sbci.metadata import format_of

    assert format_of(version) == family


@pytest.mark.parametrize(
    "version",
    [0.1, 1, "0.1", "0.1-draft", "0.1.0-final", "0.1.x.y", "0.1.0.0", "00.1.0", "0.1.0 ", "v0.1.0"],
)
def test_what_is_not_a_version_is_refused(version):
    """The float 0.1, "0.1.0-final" and "0.1.x.y" read as format 0.1, and loaded."""
    with pytest.raises(MetadataError, match="spec_version"):
        _fc(spec_version=version).validate()


def test_the_released_files_version_reads_under_a_freeze(monkeypatch):
    """A freeze to 0.1.0 keeps 0.1.0-draft files: the formats read are listed, not derived."""
    from sbci import spec

    _fc(spec_version="0.1.0-draft").validate()
    for frozen in ("0.1.0", "0.1.1"):
        monkeypatch.setattr(spec, "SPEC_VERSION", frozen)
        _fc(spec_version="0.1.0-draft").validate()
        assert template("fc")["spec_version"] == frozen
    monkeypatch.setattr(spec, "SPEC_VERSION", "1.0.0")
    _fc(spec_version="0.1.0-draft").validate()  # still listed among the formats read
    with pytest.raises(MetadataError, match="not compatible"):
        _fc(spec_version="1.0.0").validate()  # not until 1.0 is added to them


def test_the_format_written_is_one_read():
    from sbci import spec
    from sbci.metadata import format_of

    assert format_of(spec.SPEC_VERSION) in spec.READABLE_VERSIONS


def test_the_kernel_vocabulary_is_the_smoothers():
    from sbci import smoothing, spec

    assert spec.KERNELS == smoothing.KERNELS


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kernel": "gaussian"}, "kernel must be one of"),
        ({"kernel": "SHK"}, "kernel must be one of"),
        ({"bandwidth": -1}, "bandwidth must be a positive number"),
        ({"bandwidth": 0}, "bandwidth must be a positive number"),
        ({"bandwidth": "wide"}, "bandwidth must be a positive number"),
        ({"bandwidth": True}, "bandwidth must be a positive number"),
        ({"streamline_count": -5}, "streamline_count must be a whole number"),
        ({"streamline_count": 2.5}, "streamline_count must be a whole number"),
        ({"streamline_count": "many"}, "streamline_count must be a whole number"),
        ({"normalization": "none"}, "normalization must be 'unit-mass'"),
        ({"fc_nuisance_model": "24P"}, "carries fc_nuisance_model"),
        ({"mask": "none"}, "mask must be 'medial-wall'"),
        ({"area_weights": "sphere-area"}, "area_weights must be 'vertex-area'"),
    ],
)
def test_values_outside_their_domain_are_refused(sc_metadata, changes, message):
    """Each passed: a kernel there is none of, a negative bandwidth, the other modality's keys."""
    with pytest.raises(MetadataError, match=message):
        template("sc", **{**sc_metadata.fields, **changes}).validate()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kernel": "shk", "bandwidth": 0.005}, "carries kernel, bandwidth"),
        ({"streamline_count": 10}, "carries streamline_count"),
        ({"normalization": "unit-mass"}, "normalization must be 'none'"),
    ],
)
def test_an_fc_file_carries_nothing_of_sc(changes, message):
    with pytest.raises(MetadataError, match=message):
        _fc(**changes).validate()


def test_what_the_writers_write_is_in_the_domains(sc_metadata):
    """The values the build tools, the example and smooth() write all pass, a whole float too."""
    for changes in (
        {"kernel": "rdk"},
        {"kernel": "matern", "nu": 3.0},
        {"streamline_count": 0},
        {"streamline_count": 823_732.0},
        {"bandwidth": np.float32(0.005)},
        {"streamline_weighting": "unrecorded", "entry_point": "precomputed"},
        {"fc_nuisance_model": None},
    ):
        template("sc", **{**sc_metadata.fields, **changes}).validate()
    _fc(kernel=None, bandwidth=None, fc_frames=4800, fc_runs=["REST1_LR"]).validate()


def test_the_specs_constants_are_the_ones_read():
    """TRIANGLE and DTYPE reach the reader through one constant; BIDS_PATTERN, unread, is gone."""
    from sbci import spec

    assert spec.STORAGE_CONVENTION == f"{spec.TRIANGLE}-triangular-{spec.DTYPE}"
    assert template("sc")["storage_convention"] == spec.STORAGE_CONVENTION
    assert template("fc")["mask"] == spec.MASK_CONVENTION
    assert template("fc")["area_weights"] == spec.AREA_CONVENTION
    assert not hasattr(spec, "BIDS_PATTERN")

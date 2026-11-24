"""Metadata contract: the keys every SBCI file must carry, and their checks."""

from __future__ import annotations

import json
import math
import numbers
import re
from dataclasses import dataclass, field
from typing import Any

from . import spec
from .errors import SbciError


class MetadataError(SbciError, ValueError):
    """Raised when a file's metadata is missing keys or is self-inconsistent."""


_VERSION = re.compile(spec.VERSION_PATTERN)


def format_of(version: Any) -> str:
    """The format a ``spec_version`` names, ``MAJOR.MINOR`` (:mod:`sbci.spec`, the version rule).

    Raises :class:`MetadataError` for anything that is not a version: a number,
    ``"0.1"``, a tag other than ``-draft``, or more than three parts.

    Examples
    --------
    >>> format_of("0.1.0-draft"), format_of("0.1.7"), format_of("1.0.0")
    ('0.1', '0.1', '1.0')
    """
    if not isinstance(version, str):
        raise MetadataError(
            f"spec_version must be text, MAJOR.MINOR.PATCH, not the {type(version).__name__} "
            f"{version!r}"
        )
    match = _VERSION.fullmatch(version)
    if match is None:
        raise MetadataError(
            f"spec_version {version!r} is not a version: MAJOR.MINOR.PATCH, three whole "
            "numbers, with '-draft' the only tag allowed after them"
        )
    return f"{match.group(1)}.{match.group(2)}"


def _whole(value: Any) -> bool:
    """Whether ``value`` is a whole number: an integer, or a float with nothing after the point."""
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Integral):
        return True
    return isinstance(value, numbers.Real) and math.isfinite(value) and float(value).is_integer()


@dataclass
class Metadata:
    """Validated metadata for one connectome file.

    The loader refuses a file whose metadata is missing a required key, rather
    than filling in a default: a silently defaulted bandwidth or normalization
    makes two files look comparable when they are not.
    """

    fields: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a metadata key, returning ``default`` if it is absent."""
        return self.fields.get(key, default)

    @property
    def modality(self) -> str:
        """``'sc'`` or ``'fc'``."""
        return self.fields["included_connections"]

    @classmethod
    def from_json(cls, text: str | bytes) -> Metadata:
        """Parse the JSON metadata string stored alongside the connectivity."""
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise MetadataError(f"metadata is not UTF-8 text: {exc}") from exc
        try:
            # NaN and Infinity are not JSON, and to_json cannot write them back: such a
            # value reads as missing, so a required key holding one is reported missing
            loaded = json.loads(text, parse_constant=lambda _token: None)
        except json.JSONDecodeError as exc:
            raise MetadataError(f"metadata is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise MetadataError("metadata must be a JSON object")
        return cls(loaded)

    def to_json(self, indent: int | None = None) -> str:
        """Serialize back to the stored JSON string.

        NumPy scalars are written as the numbers they are, and a NaN or
        infinity is refused rather than written as the non-JSON token
        ``json.dumps`` would otherwise produce.
        """
        try:
            return json.dumps(
                self.fields, indent=indent, sort_keys=True, default=_plain, allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise MetadataError(f"metadata cannot be serialized: {exc}") from exc

    def missing_keys(self) -> list[str]:
        """Required keys absent from this metadata, in specification order.

        A key present with a ``None`` value counts as missing. Writers build
        their metadata from :func:`template`, which seeds the acquisition-
        dependent keys as ``None``; treating a null as present would let an
        unfilled template through, which is the exact failure this contract
        exists to prevent.

        Modality-specific keys are only required for their own modality: an FC
        file is not expected to declare a streamline count.
        """
        modality = self.fields.get("included_connections")
        skip: set[str] = set()
        if modality == "fc":
            skip.update(spec.SC_ONLY_KEYS)
        elif modality == "sc":
            skip.update(spec.FC_ONLY_KEYS)
        return [
            key
            for key in spec.REQUIRED_METADATA_KEYS
            if key not in skip and self.fields.get(key) is None
        ]

    def validate(self) -> None:
        """Raise :class:`MetadataError` if the metadata is unusable.

        Examples
        --------
        >>> Metadata({"grid": "fsaverage-ico4"}).validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        sbci.metadata.MetadataError: missing required metadata keys: ...
        """
        # The modality is checked first: it decides which keys are required, so
        # an unknown value must be reported as such rather than as a cascade of
        # keys the caller was never supposed to supply.
        modality = self.fields.get("included_connections")
        if modality is not None and modality not in spec.MODALITIES:
            raise MetadataError(
                f"included_connections must be one of {spec.MODALITIES}, got {modality!r}"
            )

        missing = self.missing_keys()
        if missing:
            raise MetadataError("missing required metadata keys: " + ", ".join(missing))

        grid = self.fields["grid"]
        if grid != spec.GRID:
            raise MetadataError(
                f"grid {grid!r} is not the canonical grid {spec.GRID!r}; "
                "the reduction-fraction and 4121-vertex grids are retired"
            )

        order = self.fields["hemisphere_order"]
        # a list, not anything that iterates to L then R: the string "LR" did, and a
        # number raised a TypeError that escaped SbciError
        if not isinstance(order, (list, tuple)) or tuple(order) != spec.HEMISPHERE_ORDER:
            raise MetadataError(
                f"hemisphere_order must be {list(spec.HEMISPHERE_ORDER)}, got {order!r}"
            )

        convention = self.fields["storage_convention"]
        if convention != spec.STORAGE_CONVENTION:
            raise MetadataError(
                f"storage_convention {convention!r} is not {spec.STORAGE_CONVENTION!r}; a "
                "differently stored vector would be read into the wrong matrix entries"
            )

        version = self.fields.get("spec_version")
        if version is None:
            raise MetadataError("metadata carries no spec_version")
        family = format_of(version)
        if family not in spec.READABLE_VERSIONS:
            raise MetadataError(
                f"spec_version {version!r} is not compatible with this build, which reads "
                f"format {' and '.join(sorted(spec.READABLE_VERSIONS))} and writes "
                f"{spec.SPEC_VERSION!r}"
            )

        self._check_values(modality)

        # nu, of spec.SC_OPTIONAL_KEYS, is the Matern kernel's own: smooth() writes it
        # for kernel="matern" alone, and load_cohort tells files apart by it
        nu = self.fields.get("nu")
        if nu is not None:
            kernel = self.fields.get("kernel")
            if kernel != "matern":
                raise MetadataError(
                    f"nu is the Matern kernel's smoothness, but this file's kernel is {kernel!r}"
                )
            if isinstance(nu, bool) or not isinstance(nu, numbers.Real) or not 0 < nu < math.inf:
                raise MetadataError(f"nu must be a positive number, got {nu!r}")

    def _check_values(self, modality: str) -> None:
        """Refuse values the format has no place for: the domains of the keys it fixes.

        The mask and area conventions are the format's only ones, the
        normalization is the modality's own, a key of the other modality is
        absent or empty, and an SC file's kernel, bandwidth and streamline count
        are a kernel the package knows, a positive number and a whole number.
        """
        held = (
            "a density of area-weighted mass one"
            if modality == "sc"
            else "correlations as they are"
        )
        for key, expected, meaning in (
            ("mask", spec.MASK_CONVENTION, "/mask false on the medial wall"),
            ("area_weights", spec.AREA_CONVENTION, "/area in fsaverage vertices a grid vertex"),
            ("normalization", spec.NORMALIZATION[modality], held),
        ):
            value = self.fields[key]
            if not isinstance(value, str) or value != expected:
                raise MetadataError(
                    f"{key} must be {expected!r} ({meaning}) in an {modality.upper()} file, "
                    f"got {value!r}"
                )

        others = (
            spec.FC_ONLY_KEYS if modality == "sc" else (*spec.SC_ONLY_KEYS, *spec.SC_OPTIONAL_KEYS)
        )
        carried = [key for key in others if self.fields.get(key) is not None]
        if carried:
            raise MetadataError(
                f"an {modality.upper()} file carries {', '.join(carried)}, which belong to the "
                "other modality: the metadata contradicts itself"
            )
        if modality != "sc":
            return

        kernel = self.fields["kernel"]
        if not isinstance(kernel, str) or kernel not in spec.KERNELS:
            raise MetadataError(
                f"kernel must be one of {', '.join(spec.KERNELS)}, as written, got {kernel!r}"
            )
        bandwidth = self.fields["bandwidth"]
        if (
            isinstance(bandwidth, bool)
            or not isinstance(bandwidth, numbers.Real)
            or not 0 < bandwidth < math.inf
        ):
            raise MetadataError(f"bandwidth must be a positive number, got {bandwidth!r}")
        count = self.fields["streamline_count"]
        if not _whole(count) or count < 0:
            raise MetadataError(
                f"streamline_count must be a whole number, 0 or more, got {count!r}"
            )


def _plain(value: Any) -> Any:
    """JSON-encode what the standard encoder refuses: NumPy scalars and arrays."""
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def template(modality: str, **overrides: Any) -> Metadata:
    """A metadata block with the structural keys filled and the rest empty.

    Useful for writers and for tests. Acquisition-dependent values are left as
    ``None`` on purpose so that a caller who forgets to set them gets a file
    that fails :meth:`Metadata.validate` rather than a plausible-looking lie.
    """
    if modality not in spec.MODALITIES:
        raise ValueError(f"modality must be one of {spec.MODALITIES}, got {modality!r}")

    fields: dict[str, Any] = {
        "spec_version": spec.SPEC_VERSION,
        "grid": spec.GRID,
        "hemisphere_order": list(spec.HEMISPHERE_ORDER),
        "mask": spec.MASK_CONVENTION,
        "area_weights": spec.AREA_CONVENTION,
        "storage_convention": spec.STORAGE_CONVENTION,
        "included_connections": modality,
        "normalization": None,
        "registration_reference": None,
        "pipeline_version": None,
        "container_version": None,
    }
    if modality == "sc":
        fields.update(streamline_count=None, streamline_weighting=None, kernel=None, bandwidth=None)
    else:
        fields.update(fc_nuisance_model=None)

    fields.update(overrides)
    return Metadata(fields)

r"""Reference limits for ``sbci qc``, from a folder of released files.

    python tools/build_qc_reference.py <folder> [<folder> ...] \
        --expect 946 --source "UNC Dataverse doi:10.15139/S3/NDAJYX, version 1.0" \
        --out qc_limits.json [--quantiles 0.01 0.99]

The folders hold the reference population's ``sub-<id>_sc.h5`` and
``sub-<id>_fc.h5`` files, searched with their subfolders. Every subject's
measures are taken as ``sbci qc`` takes them (:func:`sbci.qc.measures`), and
each measure's limits are its quantiles over the population, ``0.01`` and
``0.99`` unless ``--quantiles`` says otherwise, with NumPy's default (linear)
interpolation.

Limits come from the whole reference population, never from a pilot: the
tool refuses to run unless the folders hold exactly ``--expect`` subjects, so
that an unfinished download is not mistaken for the cohort. The file it
writes records where the limits came from -- ``--source``, the number of
subjects, the quantiles, the date and this package's version -- and a digest
of the files' names and sizes, which identifies the set without listing a
subject. Install it with ``$SBCI_QC_LIMITS`` or ``sbci qc --limits``; shipped
as ``src/sbci/data/qc_limits.json`` it becomes every installation's default.

A measure is compared only with files made the same way. The released young
adults' SC was smoothed by this package from the pipeline's endpoints and
their FC built from the HCP's ICA-FIX series (``tools/build_hcp_cohort.py``,
``tools/build_hcp_fc.py``), so their limits describe HCP Young Adult files
made like the release, not another dataset or the pipeline's own FSFast FC.

Reading the 946 subjects' 1,892 files takes about two hours; one subject is
held in memory at a time, about 1 GB with both files loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def find_subjects(folders: list[str]) -> dict[tuple[str, str | None], dict[str, Path]]:
    """``{(subject, session): {modality: path}}`` for every computational file in ``folders``."""
    from sbci.cohort import parse_name

    found: dict[tuple[str, str | None], dict[str, Path]] = {}
    for folder in folders:
        for path in sorted(Path(folder).rglob("*")):
            parsed = parse_name(path.name) if path.is_file() else None
            if parsed is None:
                continue
            subject, session, modality = parsed
            files = found.setdefault((subject, session), {})
            if modality in files:
                raise ValueError(
                    f"{subject} has two {modality.upper()} files: {files[modality]}, {path}"
                )
            files[modality] = path
    return found


def files_digest(subjects: dict) -> str:
    """SHA-256 over every file's name and size, one ``name<TAB>size`` line each, sorted."""
    lines = sorted(
        f"{path.name}\t{path.stat().st_size}\n"
        for files in subjects.values()
        for path in files.values()
    )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Measure every subject and write the limits."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", help="folders of the reference population's files")
    parser.add_argument(
        "--expect",
        type=int,
        required=True,
        help="how many subjects the reference population has; anything else is refused",
    )
    parser.add_argument("--source", required=True, help="where the files came from, recorded")
    parser.add_argument("--out", required=True, help="the limits file to write (.json)")
    parser.add_argument(
        "--quantiles",
        nargs=2,
        type=float,
        default=(0.01, 0.99),
        metavar=("LOW", "HIGH"),
        help="the quantiles that make the limits (default 0.01 0.99)",
    )
    args = parser.parse_args(argv)
    low_q, high_q = args.quantiles
    if not 0.0 <= low_q < high_q <= 1.0:
        parser.error(f"--quantiles must satisfy 0 <= LOW < HIGH <= 1, got {low_q} {high_q}")

    import sbci
    from sbci.qc import LIMITS_FORMAT, MEASURES, MEASURES_VERSION, measures

    subjects = find_subjects(args.folders)
    if len(subjects) != args.expect:
        print(
            f"error: the folders hold {len(subjects)} subjects, not the {args.expect} expected; "
            "limits come from the whole reference population, never from part of it",
            file=sys.stderr,
        )
        return 1
    values: dict[str, list[float]] = {name: [] for name in MEASURES}
    started = time.time()
    order = sorted(subjects, key=lambda key: (key[0], key[1] or ""))  # a session may be None
    for k, files in enumerate((subjects[key] for key in order), start=1):
        sc = sbci.load(files["sc"]) if "sc" in files else None
        fc = sbci.load(files["fc"]) if "fc" in files else None
        for name, value in measures(sc, fc).items():
            if value is not None:
                values[name].append(value)
        del sc, fc
        if k % 50 == 0 or k == len(subjects):
            print(f"{k}/{len(subjects)} subjects, {time.time() - started:.0f} s", flush=True)

    limits = {}
    for name, column in values.items():
        if not column:
            continue
        array = np.asarray(column, dtype=np.float64)
        limits[name] = {
            "low": float(np.quantile(array, low_q)),
            "high": float(np.quantile(array, high_q)),
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
            "n": int(array.size),
        }
    document = {
        "format": LIMITS_FORMAT,
        "measures_version": MEASURES_VERSION,
        "quantiles": [low_q, high_q],
        "interpolation": "linear",
        "provenance": {
            "source": args.source,
            "n_subjects": len(subjects),
            "n_sc": sum("sc" in files for files in subjects.values()),
            "n_fc": sum("fc" in files for files in subjects.values()),
            "files_digest": files_digest(subjects),
            "built": time.strftime("%Y-%m-%d", time.gmtime()),
            "sbci_version": sbci.__version__,
            "tool": "tools/build_qc_reference.py",
        },
        "limits": limits,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.name}.partial")
    partial.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, out)
    print(f"wrote limits for {len(limits)} measures from {len(subjects)} subjects to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

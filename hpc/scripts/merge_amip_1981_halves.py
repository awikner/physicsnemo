#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge per-year-fragment AMIP Zarrs into the full per-year Zarr.

The streaming converter splits a full year of files across multiple
``cpu-interactive`` Slurm jobs (each running a quarter of the 1460
files) so they fit the partition's 1h wall cap. Each job writes a
sibling ``1981_qN.zarr``. After all parts finish, this script
concatenates them along the time axis and writes the final
``1981.zarr`` consumed by the recipe + datapipe.

Usage::

    python hpc/scripts/merge_amip_1981_halves.py \\
        --parts /work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981_q1.zarr \\
                /work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981_q2.zarr \\
                /work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981_q3.zarr \\
                /work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981_q4.zarr \\
        --output /work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr

The script also accepts the legacy ``--part-a`` / ``--part-b`` flags
for the prior 2-half scheme; either form may be used.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parts",
        type=Path,
        nargs="+",
        default=None,
        help="N input fragment Zarrs to concat along the time axis, in order.",
    )
    p.add_argument(
        "--part-a", type=Path, default=None, help="Legacy 2-half form: first half."
    )
    p.add_argument(
        "--part-b", type=Path, default=None, help="Legacy 2-half form: second half."
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output if it already exists.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.parts:
        parts = list(args.parts)
    elif args.part_a and args.part_b:
        parts = [args.part_a, args.part_b]
    else:
        logger.error(
            "pass either --parts <path...> or --part-a/--part-b; got neither"
        )
        return 1

    if args.output.exists():
        if not args.overwrite:
            logger.error("%s exists; pass --overwrite to replace", args.output)
            return 1
        shutil.rmtree(args.output)

    datasets = []
    for path in parts:
        logger.info("opening %s", path)
        datasets.append(xr.open_zarr(path, consolidated=True))

    # Constants (lat, lon dims only) must match exactly; concat along
    # the time axis only.
    sizes_msg = ", ".join(f"part{i}.n_time={d.sizes['time']}" for i, d in enumerate(datasets))
    logger.info("concatenating %d parts: %s", len(datasets), sizes_msg)
    merged = xr.concat(
        datasets, dim="time", data_vars="minimal", coords="minimal"
    )
    # ``xarray.concat`` drops attrs by default; restore from the first
    # part (all parts carry the same attrs — same config / same year).
    merged.attrs = dict(datasets[0].attrs)
    merged.attrs["sample_range"] = "all"
    logger.info("merged dims: %s", dict(merged.sizes))

    logger.info("writing merged Zarr to %s", args.output)
    # Re-derive chunk encoding — preserves time_chunk=1 (the parts'
    # default) for fast random-access training reads.
    encoding = {}
    for name in merged.data_vars:
        if "time" in merged[name].dims:
            chunks = tuple(
                1 if d == "time" else merged.sizes[d] for d in merged[name].dims
            )
            encoding[name] = {"chunks": chunks}
    merged.to_zarr(
        args.output,
        mode="w",
        encoding=encoding,
        zarr_format=3,
        consolidated=True,
    )
    logger.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

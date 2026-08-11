#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract a boundary-only store at native resolution (Phase 12d option (b)).

Upstream amip_v2 trains the forecaster on a **coarse state** but feeds it
**native-resolution forcings**, which the model reduces internally with a
stride-4 convolution (``c_grid_downsample: 4``). Its fast store holds those
as two separate arrays; the fork's equivalent is two Zarr stores paired by
``ClimateZarrDataset(zarr_path=<coarse>, boundary_zarr_path=<this>)``.

Pointing ``boundary_zarr_path`` at the *full* full-resolution store works
functionally — the dataset only reads the boundary variables from it — but
would require moving the whole ~2.6 TB archive to wherever training runs.
This tool copies just the boundary variables (4 varying + 2 constant of ~49
data_vars, all 2D), which is ~2.3 GB/year instead of ~58 GB/year: a ~25x
transfer saving for a bit-identical training input.

The output carries the source's role-list attrs so the pairing self-documents,
plus ``boundary_only: true`` and the source path. NaN is **preserved** — the
runtime ``NanFillTransform`` does the masked-Gaussian coast fade
(``smooth_nan_boundaries``), which is exactly upstream's behavior and the
reason not to pre-fill here.

Usage
-----

::

    python tools/data/amip/extract_boundary_store.py \\
      --input  $ZARR_ROOT/amip_dailyavg/1981.zarr \\
      --output $ZARR_ROOT/amip_dailyavg_boundary/1981.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr
from xarray.coding.times import CFDatetimeCoder

logger = logging.getLogger("extract_boundary_store")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Full-res Zarr store.")
    p.add_argument("--output", type=Path, required=True, help="Boundary-only store.")
    p.add_argument(
        "--time-chunk",
        type=int,
        default=8,
        help="Zarr chunk length along time. Matches the coarse store's chunking "
        "so a rolling-window read touches ~one chunk per variable.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def extract_boundary_store(
    input_path: Path, output_path: Path, *, time_chunk: int = 8
) -> xr.Dataset:
    """Copy only the boundary variables of ``input_path`` into a new store."""
    src = xr.open_zarr(
        input_path, decode_times=CFDatetimeCoder(use_cftime=True), chunks=None
    )
    const = list(src.attrs.get("constant_boundary_variables", []))
    varying = list(src.attrs.get("varying_boundary_variables", []))
    names = const + varying
    if not names:
        raise ValueError(f"{input_path} lists no boundary variables in its attrs")
    missing = [n for n in names if n not in src.data_vars]
    if missing:
        raise KeyError(f"{input_path} is missing boundary variables {missing}")

    out = src[names]
    out.attrs = {
        **src.attrs,
        "boundary_only": True,
        "boundary_source_store": str(input_path),
        "boundary_note": (
            "native-resolution forcings for the upstream amip_v2 pairing: use "
            "as boundary_zarr_path alongside a coarse state store, with the "
            "model's c_grid_downsample=4. NaN is preserved on purpose — the "
            "runtime NanFillTransform applies the masked-Gaussian coast fade."
        ),
    }

    encoding = {}
    for name in names:
        dims = out[name].dims
        encoding[name] = {
            "chunks": tuple(
                min(time_chunk, out.sizes["time"]) if d == "time" else out.sizes[d]
                for d in dims
            )
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
    logger.info(
        "writing %d boundary variables (%d constant + %d varying) to %s",
        len(names), len(const), len(varying), output_path,
    )
    out.to_zarr(output_path, encoding=encoding, consolidated=True)
    return xr.open_zarr(output_path, decode_times=CFDatetimeCoder(use_cftime=True))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    extract_boundary_store(args.input, args.output, time_chunk=args.time_chunk)
    logger.info("boundary store written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

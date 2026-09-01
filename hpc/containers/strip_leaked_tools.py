# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remove the site-packages ``tools`` package that shadows this repo's own.

nvfuser, which ships inside the NGC PyTorch base image, installs a top-level
``tools`` package holding nothing but its own build helpers
(``gen_nvfuser_version.py``, ``memory.py``). Because that is a *regular* package
it wins over this repo's PEP 420 namespace ``tools/`` directories: PEP 420 gives
a regular package precedence over namespace portions found anywhere on
``sys.path``, regardless of order, so no amount of ``PYTHONPATH`` tuning helps.

The collision breaks ``from tools.data...`` and
``from tools.harmonize_hindcasts ...`` in 12 repo modules, including production
code (``tools/data/check_lat_orientation.py``,
``tools/data/plasim/extremes/nc_to_zarr.py``) and 8 test modules.

nvfuser never imports it, so it is safe to delete. Run at image build time.
"""

import importlib
import importlib.util
import pathlib
import shutil
import sys


def main() -> int:
    spec = importlib.util.find_spec("tools")
    if spec is None or not spec.origin:
        print("no regular 'tools' package on sys.path — nothing to strip")
    else:
        path = pathlib.Path(spec.origin).parent
        if "packages" not in str(path):
            print(f"'tools' resolves to {path}, which is not a site/dist-packages "
                  "directory — refusing to touch it")
            return 0
        print(f"removing leaked package: {path}")
        shutil.rmtree(path)

    # Assert the shadow is really gone, so a future base-image change that
    # reintroduces it fails the build instead of silently breaking imports.
    for mod in ("tools", "nvfuser"):
        sys.modules.pop(mod, None)
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("tools")
    if spec is not None and spec.origin and "packages" in spec.origin:
        print(f"ERROR: 'tools' is still shadowed by {spec.origin}", file=sys.stderr)
        return 1

    # nvfuser must still work without it.
    try:
        importlib.import_module("nvfuser")
    except Exception as exc:  # pragma: no cover - build-time guard
        print(f"ERROR: nvfuser broke after removing 'tools': {exc!r}", file=sys.stderr)
        return 1

    print("tools namespace clear; nvfuser still imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

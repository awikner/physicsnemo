# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Bounded async writer for xarray forecast outputs.

Mirrors the ``ThreadPoolExecutor.submit(...)`` pattern used elsewhere in
the physicsnemo examples (``examples/weather/corrdiff/generate.py``,
``examples/weather/regen/full_inference.py``) but adds three pieces the
inline pattern doesn't have:

1. **Backpressure** via a bounded semaphore — submitter blocks when more
   than ``max_in_flight`` writes are pending, so a slow disk can't let
   the in-memory queue of pending datasets grow without bound during a
   long rollout.
2. **Auto-dispatch on file extension** — ``.zarr`` → ``ds.to_zarr``,
   ``.nc`` → ``ds.to_netcdf``.
3. **Exception surfacing** — :meth:`wait_all` re-raises the first
   writer-side exception (corrdiff's pattern would silently swallow
   errors caught in ``thread.result()`` if you never look at the result).

Threading (not multiprocessing) is chosen because ``xarray.to_zarr`` /
``to_netcdf`` release the GIL during I/O, and we'd pay the IPC cost of
pickling large numpy arrays under multiprocessing.

Usage::

    with AsyncForecastWriter(max_in_flight=4) as writer:
        for ic in ic_list:
            ds = build_forecast_dataset(ic)  # xarray.Dataset
            writer.submit(ic_output_path(ic), ds)
        # __exit__ calls wait_all(): blocks until all writes flushed +
        # raises if any worker errored.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import xarray as xr


def subset_forecast_dataset(
    ds: xr.Dataset,
    *,
    surface: Optional[Sequence[str]] = None,
    upper_air: Optional[Sequence[str]] = None,
    diagnostic: Optional[Sequence[str]] = None,
    upper_air_levels: Optional[Sequence[float]] = None,
) -> xr.Dataset:
    r"""Subset a per-IC / per-chunk forecast Dataset's channel groups + levels.

    Returns a *new* Dataset (xarray's ``isel`` / ``sel`` views) where each
    ``pred_*`` data variable is filtered along its variable axis to only
    the names in the matching keyword. ``None`` (the default) keeps all
    of that group — meaning the no-arg call is the identity.

    Schema expected (matches the per-IC builder in inference.py and the
    chunked buffer in climatology_cli.py):

    - ``pred_surface``   — dims ``(.., surface_var, lat, lon)``
      with coord ``surface_var`` carrying the variable names.
    - ``pred_upper_air`` — dims ``(.., upper_air_var, level, lat, lon)``
      with coords ``upper_air_var`` and ``level``.
    - ``pred_diagnostic`` (optional) — dims ``(.., diag_var, lat, lon)``
      with coord ``diag_var``.

    Unknown names in any list raise ``KeyError`` so a typo in the config
    is loud, not silent. Passing an empty list ``[]`` is treated the same
    as ``None`` (keep all) — use the explicit ``[]`` form to *drop* the
    whole group via dropping its data variable instead (see notes below).

    Parameters
    ----------
    ds : xr.Dataset
        Per-IC / per-chunk forecast dataset.
    surface : Sequence[str], optional
        Surface variable names to keep. ``None`` keeps all.
    upper_air : Sequence[str], optional
        Upper-air variable names to keep. ``None`` keeps all.
    diagnostic : Sequence[str], optional
        Diagnostic variable names to keep. ``None`` keeps all. If the
        dataset has no ``pred_diagnostic``, this argument is ignored.
    upper_air_levels : Sequence[float], optional
        Pressure / sigma levels to keep on ``pred_upper_air``. Compared
        against the ``level`` coord by approximate equality (numpy
        ``isclose`` at default tolerance). ``None`` keeps all.

    Notes
    -----
    To *drop* an entire group (e.g. skip diagnostics entirely), pass an
    empty ``Sequence``: ``diagnostic=[]`` will drop ``pred_diagnostic``
    from the returned dataset. The same applies to the other groups —
    this lets a config say "I only want surface fields on disk".
    """
    out = ds
    if surface is not None and "pred_surface" in out:
        if len(surface) == 0:
            out = out.drop_vars("pred_surface")
        else:
            _check_known(surface, list(out["surface_var"].values), "surface")
            out = out.sel(surface_var=list(surface))
    if upper_air is not None and "pred_upper_air" in out:
        if len(upper_air) == 0:
            out = out.drop_vars("pred_upper_air")
        else:
            _check_known(upper_air, list(out["upper_air_var"].values), "upper_air")
            out = out.sel(upper_air_var=list(upper_air))
    if upper_air_levels is not None and "pred_upper_air" in out:
        if len(upper_air_levels) == 0:
            raise ValueError(
                "upper_air_levels=[] would empty pred_upper_air; pass "
                "upper_air=[] to drop the group entirely instead."
            )
        avail = np.asarray(out["level"].values, dtype=float)
        want = np.asarray(list(upper_air_levels), dtype=float)
        idx = []
        for w in want:
            matches = np.where(np.isclose(avail, w))[0]
            if matches.size == 0:
                raise KeyError(
                    f"upper_air_levels: requested level {w!r} not found in "
                    f"dataset's level coord {avail.tolist()}"
                )
            idx.append(int(matches[0]))
        out = out.isel(level=idx)
    if diagnostic is not None and "pred_diagnostic" in out:
        if len(diagnostic) == 0:
            out = out.drop_vars("pred_diagnostic")
        else:
            _check_known(diagnostic, list(out["diag_var"].values), "diagnostic")
            out = out.sel(diag_var=list(diagnostic))
    return out


def _check_known(requested: Sequence[str], available: Sequence[str], group: str) -> None:
    missing = [v for v in requested if v not in available]
    if missing:
        raise KeyError(
            f"save_variables.{group}: requested {missing!r} not in dataset's "
            f"{group}_var coord {list(available)!r}"
        )


def _write_one(path: str, dataset: xr.Dataset, *, mode: str = "w") -> None:
    """Dispatch one write based on the path's extension.

    .zarr → ``to_zarr(consolidated=True, zarr_format=3)``
    .nc  → ``to_netcdf``

    Filesystem paths only — the caller is responsible for distinct
    paths (the writer doesn't guard against two submits to the same
    location).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if str(p).endswith(".zarr"):
        dataset.to_zarr(str(p), mode=mode, zarr_format=3, consolidated=True)
    else:
        dataset.to_netcdf(str(p), mode=mode)


class AsyncForecastWriter:
    r"""Bounded-concurrency async writer for xarray Datasets.

    Parameters
    ----------
    max_in_flight : int, default 4
        Maximum number of pending (not-yet-flushed) writes. A new
        :meth:`submit` blocks once this many are outstanding so we
        don't let dataset memory grow unboundedly when the disk falls
        behind the GPU.
    num_workers : int, default 2
        Worker-thread count. Two is enough for most workloads — zarr
        write of one IC ~tens of MB is bounded by serialization, not
        throughput.
    """

    def __init__(self, *, max_in_flight: int = 4, num_workers: int = 2):
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be ≥ 1")
        if num_workers < 1:
            raise ValueError("num_workers must be ≥ 1")
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="ai_rossby_writer"
        )
        self._sem = threading.Semaphore(max_in_flight)
        self._futures: list[Future] = []
        self._futures_lock = threading.Lock()
        self._shutdown = False

    # ------------------------------------------------------------------ #
    # Context-manager API
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "AsyncForecastWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Always block on outstanding writes — even on error — so half-
        # written files don't leak past the context.
        self.wait_all()
        self._executor.shutdown(wait=True)
        self._shutdown = True

    # ------------------------------------------------------------------ #
    # Submit
    # ------------------------------------------------------------------ #

    def submit(self, path: str, dataset: xr.Dataset, *, mode: str = "w") -> Future:
        """Enqueue a write. Blocks if ``max_in_flight`` outstanding.

        Returns the underlying ``Future`` — the caller can attach a
        callback or check ``.result()`` early. The semaphore is released
        in a "done" callback so backpressure tracks completion, not
        submission.
        """
        if self._shutdown:
            raise RuntimeError("AsyncForecastWriter is shut down; submit refused")
        # Block here when too many writes are in flight.
        self._sem.acquire()
        future = self._executor.submit(_write_one, path, dataset, mode=mode)
        # Release the slot once the write actually finishes (success or fail).
        future.add_done_callback(lambda _f: self._sem.release())
        with self._futures_lock:
            self._futures.append(future)
        return future

    # ------------------------------------------------------------------ #
    # Drain
    # ------------------------------------------------------------------ #

    def wait_all(self) -> None:
        """Block until all submitted writes finish; raise on first error.

        Reaping the futures (rather than calling ``shutdown(wait=True)``
        alone) lets us surface a worker exception synchronously to the
        caller, which corrdiff's inline pattern doesn't bother with.
        """
        with self._futures_lock:
            futures = list(self._futures)
            self._futures.clear()
        first_exc: Optional[BaseException] = None
        for f in futures:
            try:
                f.result()
            except BaseException as e:  # noqa: BLE001 — re-raise after drain
                if first_exc is None:
                    first_exc = e
        if first_exc is not None:
            raise first_exc

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def in_flight(self) -> int:
        """Number of writes whose Future hasn't reaped yet."""
        with self._futures_lock:
            return sum(1 for f in self._futures if not f.done())


def make_forecast_filename(
    *,
    model_name: str,
    run_name: str,
    start_time: str,
    end_time: str,
    extension: str = "zarr",
    extra: Optional[str] = None,
) -> str:
    r"""Construct a forecast-file basename.

    Convention::

        {model_name}__{run_name}__{start_time}_{end_time}[_{extra}].{ext}

    Double underscore between the identifier sections keeps the model /
    run / time blocks unambiguously parseable even when any of them
    contains a single underscore. ``extra`` is an optional tag for
    chunk index / ensemble-member / etc.
    """
    if not extension or "." in extension:
        raise ValueError(f"extension must be a bare suffix without leading '.', got {extension!r}")
    parts = [model_name, run_name, f"{start_time}_{end_time}"]
    name = "__".join(p for p in parts if p)
    if extra:
        name = f"{name}_{extra}"
    return f"{name}.{extension}"


def format_time_for_filename(t) -> str:
    r"""Format a datetime / cftime object as ``YYYYMMDDTHHMM`` for filenames.

    Handles:

    * ``datetime.datetime`` / ``cftime.datetime`` subclasses
    * ``numpy.datetime64``
    * ISO-8601 strings (passed through, dashes/colons stripped)
    * Integer / numpy int — treated as an opaque index, formatted as ``idx{n}``

    Cftime calendars (proleptic_gregorian, 360_day, noleap) are all
    handled — the formatter only reads year/month/day/hour/minute.
    """
    # numpy datetime64 → cast to ISO + recurse
    try:
        import numpy as _np
        if isinstance(t, _np.datetime64):
            return format_time_for_filename(str(t))
    except ImportError:
        pass

    # cftime / datetime: pull the components directly
    for attr in ("year",):
        if hasattr(t, attr):
            try:
                return (
                    f"{int(t.year):04d}{int(t.month):02d}{int(t.day):02d}"
                    f"T{int(t.hour):02d}{int(t.minute):02d}"
                )
            except (AttributeError, TypeError, ValueError):
                pass

    # ISO string passthrough — strip dashes/colons.
    if isinstance(t, str):
        s = t.replace("-", "").replace(":", "").replace(" ", "T")
        if s.endswith("Z"):
            s = s[:-1]
        # Drop seconds + fractional if present (keep YYYYMMDDTHHMM)
        if "T" in s and len(s.split("T", 1)[1]) > 4:
            date, _, rest = s.partition("T")
            s = f"{date}T{rest[:4]}"
        return s

    # Fallback: opaque integer / unknown — index marker.
    return f"idx{int(t)}" if isinstance(t, int) or hasattr(t, "__int__") else "unknown"


__all__ = [
    "AsyncForecastWriter",
    "make_forecast_filename",
    "format_time_for_filename",
    "subset_forecast_dataset",
]

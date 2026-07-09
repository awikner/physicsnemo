# `tools/` — data conversion & checkpoint translation

Command-line utilities that support the ai-rossby recipe. See
[`examples/weather/ai_rossby/DATA.md`](../examples/weather/ai_rossby/DATA.md)
for the end-to-end data workflow, and
[`examples/weather/ai_rossby/PANGUWEATHER_MIGRATION.md`](../examples/weather/ai_rossby/PANGUWEATHER_MIGRATION.md)
for checkpoint conversion in context.

All scripts take explicit paths (no personal defaults) — pass `--input-dir` /
`--source-dir` / `--output`. Run each with `--help` for the full flag set.

## `tools/data/` — raw HDF5 → Zarr

Per-source converters. Each source (`e3sm`, `era5`, `plasim`, `amip`) has the
same trio:

| Script | Purpose |
|---|---|
| `<source>/pangu_h5_to_zarr.py` | per-timestep PanguWeather HDF5 → one Zarr store (per year/split) |
| `<source>/build_normalization_zarr.py` | mean/std NetCDFs → a small normalization Zarr |
| `<source>/build_climatology_zarr.py` | climatology (+ bias) store, enables ACC in validation |

Extras: `plasim/compute_delta_stats.py` (predict-delta normalization stats).
Shared helpers live in `tools/data/_common/` (`normalization.py`,
`climatology_bias.py`, `bias.py`) — imported by the converters, not run
directly. The channel-group definitions inside each `pangu_h5_to_zarr.py`
(`PANGU_<SOURCE>_CHANNELS`) **define** the variable ordering your model config
must match.

## `tools/checkpoint_translation/` — foreign checkpoints → `.mdlus`

Convert trained weights from the source stacks into PhysicsNeMo `.mdlus`:

| Script | Translates |
|---|---|
| `pangu_plasim.py` | PanguWeather `.tar` → `PanguPlasim` / `PanguPlasimLegacy` |
| `sfno_plasim.py` | PanguWeather SFNO `.tar` → `SfnoPlasim` |
| `amip_si.py` | amip Lightning `.ckpt` → diffusion wrappers (experimental) |
| `fidelity_compare.py` | numerically compare a translated model against the source |
| `_validate_translations.py` | dev harness: batch translate+load+forward-check (point `VALIDATE_*_CKPT` env vars at your checkpoints) |

Pass `--strict` to refuse writing on any missing/unexpected key — the fastest
way to catch a model-config/checkpoint mismatch. Full walkthrough:
`PANGUWEATHER_MIGRATION.md` §3.

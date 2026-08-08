# AMIP daily-avg normalization statistics

`normalize_{mean,std}_dailyavg.nc` are vendored verbatim from the upstream
[anthonyzhou-1/amip_v2](https://github.com/anthonyzhou-1/amip_v2) repo
(`norm_stats/`, baseline `e0b7b60`, Phase 12c.9). Per-variable scalar stats
plus 26-entry `level` vectors for the upper-air variables — already in the
layout `ClimateNormalizer` reads; no conversion applied.

The conversion job (`hpc/scripts/convert_amip_dailyavg_derecho.pbs`) stages
copies next to the Zarr stores so `conf/dataset/amip_dailyavg*.yaml` can
reference them under `$AI_ROSSBY_DATA`.

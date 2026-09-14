# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Channel specification for the PlaSim sim52 6-hourly emulator.

Single source of truth shared by the packer, the datapipe and the training loop.

The 52-channel prognostic state deliberately reproduces the SFNO "v11" contract so
the trained model is a drop-in for the AI-RES forecast engine, which assumes this
exact ordering. Do not reorder these lists.
"""

# --- prognostic state (52): fed back into the autoregressive loop -------------

SIGMA_VARS = ["ta", "ua", "va", "hus"]  # each on 10 sigma levels
N_SIGMA = 10

# PlaSim stores zg on 13 pressure levels; v11 uses the lowest 10 (200 hPa down).
ZG_PLEV_PA = [20000.0, 25000.0, 30000.0, 40000.0, 50000.0,
              60000.0, 70000.0, 85000.0, 92500.0, 100000.0]
ZG_NAMES = ["zg200", "zg250", "zg300", "zg400", "zg500",
            "zg600", "zg700", "zg850", "zg925", "zg1000"]

SURFACE_VARS = ["pl", "tas"]  # pl is ALREADY log surface pressure - do not re-log

STATE_CHANNELS = (
    SURFACE_VARS
    + [f"{v}{i}" for v in SIGMA_VARS for i in range(1, N_SIGMA + 1)]
    + ZG_NAMES
)
assert len(STATE_CHANNELS) == 52, len(STATE_CHANNELS)

# --- forcing (9): input only, never predicted --------------------------------
# lsm/sg/z0 are static; sst/sic/rsdt are a repeating annual cycle; the last three
# are computed analytically. `sg` IS orography (surface geopotential = g * height),
# so no separate orography channel is added.

STATIC_FORCING = ["lsm", "sg", "z0"]
CYCLIC_FORCING = ["sst", "rsdt"]
COMPUTED_FORCING = ["cos_zenith", "lat_grid", "lon_grid"]
FORCING_CHANNELS = STATIC_FORCING + CYCLIC_FORCING + COMPUTED_FORCING
assert len(FORCING_CHANNELS) == 8, len(FORCING_CHANNELS)

# NOTE: v11's 6th forcing `sic` is currently ABSENT here, and that is a MISTAKE
# preserved only because the trained checkpoints depend on this channel count.
#
# sic is stored as a masked array whose `.data` is all-NaN: the information is in
# the MASK, which marks where sea ice is. Filling masked entries with the field's
# valid mean (as the packer does for `sst`) collapses it to a constant 1.0, which
# is what led to the wrong conclusion that it was uninformative. The correct
# treatment is mask -> 0, giving a binary sea-ice indicator:
#     mask->binary  mean 0.12498  std 0.33070
#     v11 reference mean 0.12497  std 0.33068   (exact match)
#
# To restore it: special-case sic in tools/pack_plasim.py (_fill_masked -> 0),
# append "sic" to CYCLIC_FORCING, repack the forcing files only (cheap - they are
# ~363 MB/year and derived), and either retrain or widen the encoder's input
# convolution by one zero-initialised channel and fine-tune.

# --- diagnostics (2): output only, NEVER fed back ----------------------------
# In the RAW sim52 files pr_6h is an ACCUMULATION in m/6h (max ~0.022 -> ~89 mm/day
# via *1000*4, matching AI-RES scores/pipeline.py::_extract_pr). The separate
# m/s-rate convention applies to the emulator's *output* channel at the AI-RES
# boundary (PanguPlasimFS.py:204-230) - do not conflate the two.
#
# evap is stored NON-POSITIVE (<= 0, upward flux convention); we model -evap >= 0.
# Both diagnostics are zero-inflated (pr_6h ~31% exact zeros, -evap ~10%), so both
# use the same hurdle/censored-Gamma family rather than a Gaussian.

DIAGNOSTIC_CHANNELS = ["pr_6h", "evap"]
PRECIP_CHANNEL = "pr_6h"
# Diagnostics stored as non-negative quantities: pr_6h as-is, evap negated.
DIAGNOSTIC_SIGN = {"pr_6h": 1.0, "evap": -1.0}

# The two diagnostics are NOT in the same units, which is easy to get wrong when
# reporting: pr_6h is an accumulation in m/6h, evap is a rate in m/s. Multiply by
# these to get mm/day. Training is unaffected (the Gamma scale is fitted per
# channel), but any physical-units metric or plot must use them.
DIAGNOSTIC_TO_MM_PER_DAY = {
    "pr_6h": 1000.0 * 4.0,       # m/6h  -> mm/day
    "evap": 1000.0 * 86400.0,    # m/s   -> mm/day
}

# Distribution parameters emitted per diagnostic in probabilistic mode.
# Each diagnostic -> hurdle/censored-Gamma (p, k, theta).
DIAG_PARAM_NAMES = ["pr_p", "pr_k", "pr_theta", "evap_p", "evap_k", "evap_theta"]

N_STATE = len(STATE_CHANNELS)              # 52
N_FORCING = len(FORCING_CHANNELS)          # 8
N_DIAGNOSTIC = len(DIAGNOSTIC_CHANNELS)    # 2
N_DIAG_PARAMS = len(DIAG_PARAM_NAMES)      # 6

# Model I/O:
#   deterministic:  in = 52 + 8 + n_noise, out = 52 + 2 = 54
#   probabilistic:  in = 52 + 8 + n_noise, out = 52 + 6 = 58

GRID_SHAPE = (64, 128)  # T42 Gaussian
DHOURS = 6

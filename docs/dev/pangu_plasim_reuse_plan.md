# Pangu_Plasim ↔ `physicsnemo.nn` component-reuse plan

Companion to [implementation_plan.md](implementation_plan.md). This document
records the analysis and three-phase plan for replacing the locally vendored
PanguWeather building blocks under
`physicsnemo/experimental/models/pangu_plasim/{_pangu_utils.py, layers.py}`
with the equivalent reusable components shipped in `physicsnemo.nn`, while
preserving the **faithful** flavor's checkpoint and numerical fidelity to
PanguWeather v2.0.

Both `PanguPlasim` and `PanguPlasimLegacy` share the same building blocks
(`_pangu_utils.py` + `layers.py`), so each phase applies to both models
simultaneously.

---

## 1. Component compatibility (PanguWeather v2.0 vs physicsnemo.nn)

| Component | Local file | `physicsnemo.nn` equivalent | Identical to v2.0 source? | Decision |
|---|---|---|---|---|
| `PatchEmbed2D` / `PatchEmbed3D` | `_pangu_utils.py` | `physicsnemo.nn.module.utils.PatchEmbed{2D,3D}` | Pending pointwise verification (both derived from v2.0) | **Swap** (Phase A) |
| `PatchRecovery2D` / `PatchRecovery3D` | `_pangu_utils.py` | `physicsnemo.nn.module.utils.PatchRecovery{2D,3D}` | Pending pointwise verification | **Swap** (Phase A) |
| `crop2d`, `crop3d` | `_pangu_utils.py` | `physicsnemo.nn.module.utils.utils` | Likely identical | **Swap** (Phase A) |
| `get_pad2d`, `get_pad3d` | `_pangu_utils.py` | `physicsnemo.nn.module.utils.utils` | Pending pointwise verification | **Swap** (Phase A) |
| `get_earth_position_index` | `_pangu_utils.py` | `physicsnemo.nn.module.utils.utils` | Pending pointwise verification | **Swap** (Phase A) |
| `window_partition`, `window_reverse` | `_pangu_utils.py` | `physicsnemo.nn.module.utils.shift_window_mask` | Upstream adds backward-compatible `ndim=3` kwarg | **Swap** (Phase A) |
| **`get_shift_window_mask`** | `_pangu_utils.py` (cyclic-correct, faithful to v2.0) | `physicsnemo.nn.module.utils.shift_window_mask` | ❌ **Issue [#1599](https://github.com/NVIDIA/physicsnemo/issues/1599)** — physicsnemo partitions the longitude axis (creating 27 region IDs in 3D / 9 in 2D) instead of leaving it unpartitioned (9 / 3) for the cyclic dateline behavior Pangu-Weather requires. Cross-dateline attention is suppressed. | **Vendor + fix locally** (Phase C). PR fix upstream; drop vendor once merged. |
| `DownSample3D` | `layers.py` (parametrized `updown_scale_factor`) | `physicsnemo.nn.DownSample3D` (hard-codes factor=2) | ✅ for `factor=2` only | **Conditional swap** (Phase B) — branch on `factor == 2` |
| `UpSample3D` | `layers.py` (parametrized) | `physicsnemo.nn.UpSample3D` (hard-codes factor=2) | ✅ for `factor=2` only | **Conditional swap** (Phase B) |
| `Mask` | `layers.py` | none | Pangu_Plasim-specific (land/ocean output masking) | **Keep local** indefinitely |
| `Mlp` | `layers.py` | `physicsnemo.nn.module.mlp_layers.Mlp` | Pending pointwise verification | **Swap** (Phase C, with the block) |
| `EarthAttention3D` | `layers.py` (uses `F.scaled_dot_product_attention`) | `physicsnemo.nn.module.attention_layers.EarthAttention3D` (explicit `q @ k.T → softmax → attn @ v`) | ⚠ mathematically equivalent at `attn_drop=0`; not bit-identical (~1e-5–1e-4 drift from SDPA kernel selection) | **Vendor (Phase C)** with the SDPA path. Open upstream issue / PR for an opt-in `use_sdpa` kwarg, then swap. |
| **`EarthSpecificBlock`** ↔ `Transformer3DBlock` | `layers.py` (has `vertical_windowing` flag; uses faithful cyclic mask) | `physicsnemo.nn.module.transformer_layers.Transformer3DBlock` (always-vertical-shift; uses buggy mask) | ❌ uses buggy mask; missing `vertical_windowing=False` mode | **Vendor + fix locally** (Phase C). PR upstream both fixes; drop vendor once merged. |
| **`EarthSpecificLayer`** ↔ `FuserLayer` | `layers.py` | `physicsnemo.nn.module.transformer_layers.FuserLayer` | ❌ built on the broken block | **Vendor + fix locally** (Phase C) |

State-dict key compatibility note: physicsnemo's `Transformer3DBlock` and `FuserLayer` use the same submodule names (`norm1`, `attn`, `drop_path`, `norm2`, `mlp`, `blocks`) as our `EarthSpecificBlock` / `EarthSpecificLayer`. Renaming the classes preserves checkpoint keys.

---

## 2. Phased plan

### Phase A — Safe utility swaps (narrowed scope)

**Scope:** swap **pure tensor utilities** to `physicsnemo.nn.module.utils`:

- `crop2d`, `crop3d`, `get_pad2d`, `get_pad3d`, `get_earth_position_index`
- `window_partition`, `window_reverse`

**Deferred to follow-up (A.5):** `PatchEmbed2D`, `PatchEmbed3D`,
`PatchRecovery2D`, `PatchRecovery3D`. Reasons:

- Our local `PatchEmbed2D` / `PatchEmbed3D` set extra attributes
  (`output_size`, `padded_front`) that the model classes read in
  `__init__`. Swapping requires moving those computations to the model side
  or wrapping the physicsnemo class.
- The local `PatchEmbed3D` carries a faithful bug from the PanguWeather
  v2.0 source: when `l_remainder != 0`, ``padding_front = l_pad -
  padding_front`` uses ``padding_front=0`` initial value and ends up swapping
  the front/back split vs. physicsnemo's correct ``padding_front = l_pad //
  2``. The output sum is identical (so output shape is identical), but the
  asymmetry direction flips. `padded_front` propagates to
  `SubPixelConvICNR_3D` recovery heads. This is bit-identical for our
  current smoke config (8 levels, `l_remainder == 0`) but would change
  behavior for canonical ERA5 configs (13 levels, `subpixel_deconv=True`).
  Defer until we can decide whether to: (a) preserve the source's behavior
  by keeping local `PatchEmbed3D`, (b) follow physicsnemo's correct path
  and patch the SubPixelConv recovery code accordingly, or (c) PR
  `padded_front` exposure to upstream.

**Test gate:** add `test/models/pangu_plasim/test_utils_equivalence.py` with
one `torch.equal`-level assertion per swapped util. Failures here block
Phase A.

**Cleanup:** in `_pangu_utils.py`, remove the functions/classes that no longer
have any local consumer. What stays in `_pangu_utils.py` after Phase A:

- `PatchEmbed{2D,3D}_Cyclic` (no upstream equivalent)
- `SubPixelConvICNR_{2D,3D}[_wHead]` (no upstream equivalent)
- `PatchRecovery5` (no upstream equivalent)
- `PolarPad{2d,3d}`, `Interpolate`, `Integrator`, `ICNR` (no upstream equivalent)
- `get_shift_window_mask` (vendored — Phase C touches this)

**Acceptance:** existing MOD-008b non-regression + Delta smoke pass unchanged.

### Phase B — Down/Up sample swaps (factor=2 path)

**Scope:** in `layers.py`, branch `DownSample.__init__` / `UpSample.__init__` on
`downsample_factor == 2`. When 2, delegate to `physicsnemo.nn.DownSample3D` /
`UpSample3D`. Otherwise fall back to the local parametrized implementation. Same
public API.

**Test gate:** existing `test_pangu_plasim_non_regression` must remain green
(the smoke config uses `updown_scale_factor=2`, so this exercises the swapped
path). Cross-check the local-fallback path with a separate
parametrized test variant at `updown_scale_factor=3` to verify the branch.

**Upstream contribution:** open a PR adding a `factor` kwarg to
`physicsnemo.nn.DownSample3D` / `UpSample3D` (defaults to 2 — backward
compatible). Once merged, drop the local fallback.

### Phase C — Block / layer swap (vendor with the mask fix)

**Scope:** create a new sub-package alongside `layers.py`:

```
physicsnemo/experimental/models/pangu_plasim/
├── _vendored_physicsnemo_nn/
│   ├── __init__.py
│   ├── attention_layers.py       # EarthAttention3D + opt-in use_sdpa
│   ├── transformer_layers.py     # Transformer3DBlock (+ vertical_windowing), FuserLayer
│   └── shift_window_mask.py      # get_shift_window_mask with #1599 fix
└── layers.py                     # Mask only after this phase; re-exports vendored classes
```

The vendored files **copy** the corresponding upstream sources and apply the
following patches:

1. **`shift_window_mask.py`** — drop the longitude loop. In 3D, iterate only over
   `pl_slices × lat_slices`; in 2D, only over `lat_slices`. Fill `img_mask` of
   shape `(1, Pl, Lat, Lon, 1)` (3D) or `(1, Lat, Lon, 1)` (2D) at
   `[:, pl, lat, :, :] = cnt` (3D) or `[:, lat, :, :] = cnt` (2D). 9 region
   IDs (3D) / 3 region IDs (2D). **This matches the PanguWeather v2.0 source and
   fixes [#1599](https://github.com/NVIDIA/physicsnemo/issues/1599).**

2. **`transformer_layers.py`** —
   - Add `vertical_windowing: bool = True` kwarg to `Transformer3DBlock.__init__`.
   - When `False`, force `shift_size = (0, w_lat // 2, w_lon // 2)` and gate
     `self.roll` on `shift_lat and shift_lon` only.
   - Pass `vertical_windowing` through `FuserLayer.__init__` to each block.

3. **`attention_layers.py`** —
   - Add `use_sdpa: bool = False` kwarg to `EarthAttention3D.__init__`.
   - When `True`, route through `F.scaled_dot_product_attention` (matching our
     local implementation); default `False` preserves bit-identical behavior
     with the existing upstream model.

`layers.py` after Phase C: keeps `Mask` only, re-exports `FuserLayer`,
`Transformer3DBlock`, `EarthAttention3D`, `Mlp` from `_vendored_physicsnemo_nn/`.
Model code is updated to import `FuserLayer` from the vendored sub-package
(instead of `EarthSpecificLayer` from `layers.py`). Model attribute names
(`self.layer1`, `self.downsample`, etc.) and submodule names inside the blocks
(`norm1`, `attn`, `mlp`, …) are unchanged, so **state_dict keys are
preserved** — translated PanguWeather checkpoints still load.

**Tests added in Phase C:**

- `test/models/pangu_plasim/test_shift_window_mask.py` — exact value
  comparison of the fixed `get_shift_window_mask` against an independently
  computed reference for a tiny `input_resolution`. Verifies 9 region IDs
  (3D) / 3 IDs (2D).
- `test/models/pangu_plasim/test_block_equivalence.py` — pointwise
  comparison of the vendored `Transformer3DBlock` against our existing
  `EarthSpecificBlock` for the same inputs/weights/seed. Should be
  bit-identical (since both algorithms are equivalent and use the same
  fixed mask).
- Re-run `test_pangu_plasim_non_regression` and the Delta smoke test —
  must remain green at the existing tolerance.

**Intentional deviation note (#1599 fix).** The Phase C swap means the faithful
`PanguPlasim` / `PanguPlasimLegacy` reproduce PanguWeather v2.0's *intent*
(cross-dateline attention) but technically deviate from physicsnemo's pre-fix
`Pangu` behavior. This deviation:

- Is documented in this file and in the vendored `shift_window_mask.py` header.
- Will disappear once the upstream `#1599` PR is merged, at which point we drop
  the vendor and import directly from `physicsnemo.nn`.
- Does not break our own checkpoint compatibility — the mask is registered as a
  non-trainable buffer; mask shape (and therefore the trained weights' usable
  receptive field) was correct in PanguWeather v2.0 to begin with.

**Upstream contribution path:**

1. Open PR(s) on `NVIDIA/physicsnemo`:
   - Fix `#1599` — `get_shift_window_mask` longitude treatment.
   - Add `vertical_windowing` kwarg to `Transformer3DBlock` / `FuserLayer`.
   - Add `use_sdpa` opt-in to `EarthAttention3D`.
2. Once merged in a release that lands on Delta's torch/Python pin, replace the
   vendored sub-package with direct imports from `physicsnemo.nn` and delete
   `_vendored_physicsnemo_nn/`.

---

## 3. Open decisions for future phases

- **`SubPixelConv*` family + `PolarPad*` + `Interpolate` + `Integrator`** — no
  upstream equivalents. Candidates for an eventual PR to
  `physicsnemo.nn.module.utils` / `physicsnemo.nn.module.resample_layers`. Out
  of scope for this plan.
- **`PatchEmbed{2D,3D}_Cyclic`** — same; could be folded into upstream
  `PatchEmbed*` as a `cyclic: bool = False` kwarg in a follow-up PR.

---

## 4. Cross-references

- Implementation plan: [implementation_plan.md](implementation_plan.md)
- Coding standards: [CODING_STANDARDS/MODELS_IMPLEMENTATION.md](CODING_STANDARDS/MODELS_IMPLEMENTATION.md)
- Delta recipe: [hpc/delta.md](hpc/delta.md)
- Upstream issue: [physicsnemo#1599](https://github.com/NVIDIA/physicsnemo/issues/1599)

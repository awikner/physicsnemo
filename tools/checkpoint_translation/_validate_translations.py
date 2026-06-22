#!/usr/bin/env python
"""Translate + load + forward-check all three user-provided PanguWeather checkpoints."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import torch

# silence the experimental-namespace warning during the heavy imports
warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")

sys.path.insert(0, "/work/nvme/bdiu/awikner/physicsnemo/tools/checkpoint_translation")
import pangu_plasim as pp_trans
import sfno_plasim as sfno_trans


CASES = [
    {
        "label": "PanguPlasimLegacy (PLASIM 0515)",
        "translator": pp_trans,
        "ckpt": "/work/nvme/bdiu/awikner/PanguWeather/v2.0/results/PLASIM/0515/checkpoints/ckpt_epoch_99.tar",
        "yaml": "/work/nvme/bdiu/awikner/physicsnemo/examples/weather/ai_rossby/conf/model/pangu_plasim_legacy.yaml",
        "target_class": "PanguPlasimLegacy",
        # 13 pressure levels × 5 upper-air vars, 64x128
    },
    {
        "label": "PanguPlasimLegacy (S2S, mapped from user's 'PanguPlasim' label)",
        "translator": pp_trans,
        "ckpt": "/work/hdd/bdiu/awikner/PanguWeather-rajatm2/v2.0/results/S2S/2000/training_checkpoints/best_ckpt.tar",
        "yaml": "/work/nvme/bdiu/awikner/physicsnemo/examples/weather/ai_rossby/conf/model/pangu_plasim_s2s.yaml",
        "target_class": "PanguPlasimLegacy",
        # 17 pressure levels × 5 upper-air vars, 180x360
    },
    {
        "label": "SfnoPlasim (PLASIM 5412)",
        "translator": sfno_trans,
        "ckpt": "/work/nvme/bdiu/awikner/PanguWeather/v2.0/results/SFNO/5412/checkpoints/ckpt_epoch_25.tar",
        "yaml": "/work/nvme/bdiu/awikner/physicsnemo/examples/weather/ai_rossby/conf/model/sfno_plasim_5412.yaml",
        "target_class": "SfnoPlasim",
        # 10 levels, 64x128
    },
]


def _make_inputs(cfg_yaml: Path, device: torch.device) -> dict:
    """Build random inputs of the right shape based on the model YAML."""
    import yaml as _yaml
    with open(cfg_yaml) as fh:
        cfg = _yaml.safe_load(fh)
    H, W = cfg["horizontal_resolution"]
    n_s = len(cfg["surface_variables"])
    n_l = len(cfg.get("land_variables", []) or [])
    n_o = len(cfg.get("ocean_variables", []) or [])
    n_c = len(cfg["constant_boundary_variables"])
    n_v = len(cfg["varying_boundary_variables"])
    n_u = len(cfg["upper_air_variables"])
    n_L = len(cfg["levels"])
    # patchembed2d's in_chans = num_surface + num_land + num_ocean +
    # num_constant + num_varying — so surface_in must pack the surface
    # + land + ocean channels together. The model's input-validation
    # check at pangu_plasim_legacy.py:558 ignores the +num_land/+num_ocean
    # contribution and incorrectly demands num_surface; we bypass that
    # check via the torch.compiler.is_compiling monkey-patch in run_one().
    return {
        "surface": torch.randn(1, n_s + n_l + n_o, H, W, device=device),
        "constant_boundary": torch.randn(1, n_c, H, W, device=device),
        "varying_boundary": torch.randn(1, n_v, H, W, device=device),
        "upper_air": torch.randn(1, n_u, n_L, H, W, device=device),
    }


def _bypass_validation_ctx():
    """Context manager that makes torch.compiler.is_compiling() return True.

    PanguPlasimLegacy.forward gates its input-shape validation behind
    ``if not torch.compiler.is_compiling()`` — flipping that to True
    lets us run the forward with the *correct* channel layout (surface
    + land + ocean packed together) that the model's inner patchembed2d
    actually expects, without tripping the over-strict outer check.
    """
    import contextlib
    @contextlib.contextmanager
    def _ctx():
        orig = torch.compiler.is_compiling
        torch.compiler.is_compiling = lambda: True
        try:
            yield
        finally:
            torch.compiler.is_compiling = orig
    return _ctx()


def _summarize_tensor(t):
    """Return (min, max, mean, std, frac_nonzero) for a tensor."""
    arr = t.detach().float()
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "finite": bool(torch.isfinite(arr).all()),
        "frac_nonzero": float((arr.abs() > 1e-10).float().mean()),
    }


def run_one(case: dict, device: torch.device) -> dict:
    label = case["label"]
    print(f"\n{'='*70}")
    print(f"== {label}")
    print(f"{'='*70}")
    print(f"  checkpoint = {case['ckpt']}")
    print(f"  yaml       = {case['yaml']}")
    print(f"  target     = {case['target_class']}")

    # 1) Load source state dict.
    t0 = time.perf_counter()
    src_sd = case["translator"].load_panguweather_state_dict(Path(case["ckpt"]))
    print(f"  source state dict: {len(src_sd)} tensors (in {time.perf_counter()-t0:.1f}s)")

    sample_key = list(src_sd.keys())[0]
    print(f"  first source key  = {sample_key}")
    has_module = sum(1 for k in src_sd if k.startswith("module.")) > 0
    has_orig = sum(1 for k in src_sd if k.startswith("_orig_mod.") or "._orig_mod." in k) > 0
    print(f"  DDP-wrapped (module. prefix): {has_module}")
    print(f"  torch.compile-wrapped (_orig_mod): {has_orig}")

    # 2) Translate.
    t0 = time.perf_counter()
    tgt_sd = case["translator"].translate_state_dict(src_sd)
    print(f"  translated: {len(tgt_sd)} tensors (in {time.perf_counter()-t0:.1f}s)")
    print(f"  first translated key = {list(tgt_sd.keys())[0]}")

    # 3) Build target + load state.
    t0 = time.perf_counter()
    try:
        kw = {}
        if "target_class" in case["translator"].build_target_model_from_yaml.__code__.co_varnames:
            kw["target_class"] = case["target_class"]
        model = case["translator"].build_target_model_from_yaml(Path(case["yaml"]), **kw)
    except TypeError:
        # SFNO translator doesn't take target_class
        model = case["translator"].build_target_model_from_yaml(Path(case["yaml"]))
    print(f"  built {type(model).__name__} (in {time.perf_counter()-t0:.1f}s)")

    res = model.load_state_dict(tgt_sd, strict=False)
    print(f"  load_state_dict: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        print(f"    sample missing: {res.missing_keys[:5]}")
    if res.unexpected_keys:
        print(f"    sample unexpected: {res.unexpected_keys[:5]}")

    # 4) Forward pass on GPU.
    print(f"  moving to {device} + forward pass …")
    model = model.to(device).eval()
    inp = _make_inputs(Path(case["yaml"]), device)
    t0 = time.perf_counter()
    with torch.no_grad(), _bypass_validation_ctx():
        out = model(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
    print(f"  forward done in {time.perf_counter()-t0:.2f}s")

    # 5) Inspect outputs.
    surface_pred, upper_pred = out[0], out[1]
    diag_pred = out[2] if isinstance(out[2], torch.Tensor) and out[2].numel() > 1 else None
    print(f"  surface output shape: {tuple(surface_pred.shape)}")
    print(f"    surface stats: {_summarize_tensor(surface_pred)}")
    print(f"  upper_air output shape: {tuple(upper_pred.shape)}")
    print(f"    upper_air stats: {_summarize_tensor(upper_pred)}")
    if diag_pred is not None:
        print(f"  diagnostic output shape: {tuple(diag_pred.shape)}")
        print(f"    diagnostic stats: {_summarize_tensor(diag_pred)}")

    # 6) Sanity assertions.
    finite_ok = torch.isfinite(surface_pred).all() and torch.isfinite(upper_pred).all()
    nonzero_ok = (surface_pred.abs() > 1e-8).any() and (upper_pred.abs().mean() > 1e-8)
    return {
        "label": label,
        "missing": len(res.missing_keys),
        "unexpected": len(res.unexpected_keys),
        "finite": bool(finite_ok),
        "nonzero": bool(nonzero_ok),
        "surface_stats": _summarize_tensor(surface_pred),
        "upper_stats": _summarize_tensor(upper_pred),
        "diag_stats": _summarize_tensor(diag_pred) if diag_pred is not None else None,
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    results = []
    for case in CASES:
        try:
            results.append(run_one(case, device))
        except Exception as e:
            print(f"FAILED {case['label']}: {type(e).__name__}: {e}")
            results.append({"label": case["label"], "error": str(e)})

    print(f"\n\n{'='*70}\n== SUMMARY\n{'='*70}")
    for r in results:
        if "error" in r:
            print(f"  ✗ {r['label']}: {r['error']}")
        else:
            ok = r["missing"] == 0 and r["unexpected"] == 0 and r["finite"] and r["nonzero"]
            mark = "✓" if ok else "⚠"
            print(
                f"  {mark} {r['label']}\n"
                f"      missing={r['missing']}, unexpected={r['unexpected']}, "
                f"finite={r['finite']}, nonzero={r['nonzero']}\n"
                f"      surface: mean={r['surface_stats']['mean']:.4e}, "
                f"std={r['surface_stats']['std']:.4e}, "
                f"max|={max(abs(r['surface_stats']['min']), abs(r['surface_stats']['max'])):.4e}"
            )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-epoch training step + optimizer/scheduler factories for Pangu_Plasim.

Phase 3 v1: PanguPlasimLegacy (deterministic, no VAE-KL). The optimizer +
scheduler choices come from PanguWeather v2.0 config conventions
(AdamW + OneCycleLR for the legacy variant; AdamW + LinearWarmupCosineAnnealingLR
for the future PanguPlasim with VAE).
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model step in store rows (2026-08-13 stride audit) — one resolution point.
# ---------------------------------------------------------------------------


def model_step_rows(cfg: Any, dataset: Any) -> int:
    """Store rows advanced per model step, for this cfg's model/dataset pair.

    The step is a property of the **model family**, not of the store — the
    PLASIM ``sim52`` archive feeds ``pangu_plasim*`` (24 h, upstream
    ``PANGU_PLASIM_H5_DERECHO_0514.yaml``) and ``sfno_plasim*`` (6 h,
    ``SFNO_PLASIM_H5_DERECHO_5412.yaml``) from the same 6-hourly rows. So
    ``cfg.model.timedelta_hours`` is authoritative; the dataset's
    ``forecast_lead_times`` (rows) and optional ``timedelta_hours`` are
    cross-checked against it and a disagreement raises.

    Every driver — training loaders, rollout validation, inference, eval —
    resolves the step through here, because a stride mismatch changes no shape
    and yields a healthy-looking loss.
    """
    from physicsnemo.experimental.datapipes.climate import resolve_step_stride

    data_cfg = cfg.get("dataset", None)
    model_cfg = cfg.get("model", None)
    leads = data_cfg.get("forecast_lead_times", None) if data_cfg else None
    return resolve_step_stride(
        dataset,
        forecast_lead_times=list(leads) if leads else None,
        timedelta_hours=data_cfg.get("timedelta_hours", None) if data_cfg else None,
        model_timedelta_hours=(
            model_cfg.get("timedelta_hours", None) if model_cfg else None
        ),
    )


def model_step_hours(cfg: Any, dataset: Any) -> float:
    """Wall-clock hours advanced per model step.

    ``model_step_rows`` in units the physical world uses: rows-per-step times
    the store's own ``data_timedelta_hours``. Anything that needs to convert a
    duration into a step count — bin widths, horizons — goes through here, so
    the conversion is derived from the same single source of truth as the
    stride rather than hard-coded per config.
    """
    rows = model_step_rows(cfg, dataset)
    layout = getattr(dataset, "layout", None)
    hours_per_row = float(getattr(layout, "data_timedelta_hours", 0) or 0)
    if hours_per_row <= 0:
        raise ValueError(
            "the store declares no data_timedelta_hours, so a duration cannot "
            "be converted to model steps; set the bin widths explicitly"
        )
    return rows * hours_per_row


#: Mean length of a calendar month, in days (365.25 / 12). The bin widths this
#: converts are aggregation windows, not calendar arithmetic — a fixed mean
#: month is both sufficient and what makes every bin the same size.
_HOURS_PER_MONTH = 365.25 / 12.0 * 24.0


def steps_per_month(cfg: Any, dataset: Any) -> int:
    """Model steps in one mean calendar month, at this run's timestep.

    24-hour step over 6-hourly rows -> 30; a 6-hour step -> 122. The shipped
    eval-suite defaults used to hard-code the 6-hourly number (120) and label
    it "≈ 1 month", which at the AMIP 24-hour step silently made every
    "monthly" bin four months wide.
    """
    return max(1, round(_HOURS_PER_MONTH / model_step_hours(cfg, dataset)))


def lead_times_for_sampler(cfg: Any, step_rows: int) -> list[int]:
    """The single-step pair's lead list, in store rows.

    ``cfg.dataset.forecast_lead_times`` when it is set; otherwise the model's
    own step. Stores shared by families with different steps leave it null so
    the model supplies it — that is what keeps a shared PLASIM config from
    having to be wrong for one of them.
    """
    data_cfg = cfg.get("dataset", None)
    leads = data_cfg.get("forecast_lead_times", None) if data_cfg else None
    return [int(v) for v in leads] if leads else [int(step_rows)]



# ---------------------------------------------------------------------------
# Predicted ocean channels (Phase 12f) — one injection point.
# ---------------------------------------------------------------------------


def adopt_ocean_contract(scheduler: Any, model: torch.nn.Module) -> Any:
    """Teach a diffusion scheduler the model's predicted-ocean contract.

    ``nocean`` (how many tail channels of the state axis are ocean) and
    ``ocean_grid_indices`` (which ``c_grid`` channels their truth is read
    from) are two halves of one fact, and the model already derives both from
    ``ocean_state_variables``. Every driver — training, mid-training
    validation, ``inference.py``, ``eval_diffusion.py`` — routes through this
    function instead of restating them in a config, because a scheduler built
    without them does not pad, impose, or supervise the ocean block: the model
    would then be handed a state-width window and fail on a channel count, or
    worse, run with the ocean block left at whatever the sampler happened to
    put there.

    A no-op for models without ocean channels and for schedulers that have no
    ocean support (the single-step SI / SI_X families).
    """
    nocean = int(getattr(model, "num_ocean", 0) or 0)
    if not nocean or not hasattr(scheduler, "nocean"):
        return scheduler
    indices = list(getattr(model, "ocean_grid_indices", []))
    if len(indices) != nocean:
        raise ValueError(
            f"model reports num_ocean={nocean} but "
            f"ocean_grid_indices={indices}; the two are derived together and "
            f"must agree"
        )
    scheduler.nocean = nocean
    scheduler.ocean_grid_indices = indices
    logger.info(
        f"ocean contract: nocean={nocean}, c_grid indices={indices} "
        f"({type(scheduler).__name__})"
    )
    return scheduler



# ---------------------------------------------------------------------------
# Partial-checkpoint warm start (Phase 12f) — upstream amip's
# ``load_partial_weights``.
# ---------------------------------------------------------------------------

#: Wrapper prefixes torch adds around the real module tree, stripped so a
#: checkpoint saved under DDP or ``torch.compile`` warm-starts a plain model.
_WRAP_PREFIXES = ("module.", "_orig_mod.", "model.")


def _strip_wrap_prefixes(key: str) -> str:
    """Iteratively drop DDP / compile / Lightning wrapper prefixes."""
    changed = True
    while changed:
        changed = False
        for prefix in _WRAP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _mdlus_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Read ``model.pt`` out of a ``.mdlus`` archive (zip or legacy tar).

    Deliberately *not*
    :meth:`physicsnemo.core.module.Module.from_checkpoint`: that would
    instantiate the source model from its stored args — the pre-warm-start
    architecture, several GB of parameters we would immediately discard —
    and then still refuse a shape mismatch. We only ever want its tensors.
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as archive:
            blob = archive.read("model.pt")
        return torch.load(io.BytesIO(blob), map_location="cpu")
    with tarfile.open(path, "r") as tar:
        member = tar.extractfile("model.pt")
        if member is None:
            raise IOError(f"{path} has no 'model.pt' member")
        return torch.load(io.BytesIO(member.read()), map_location="cpu")


def _extract_state_dict(blob: Any) -> dict[str, torch.Tensor]:
    """Pull a flat ``{name: tensor}`` mapping out of a loaded checkpoint."""
    if isinstance(blob, dict):
        for key in ("state_dict", "model_state_dict"):
            inner = blob.get(key)
            if isinstance(inner, dict):
                return inner
    if not isinstance(blob, dict):
        raise TypeError(f"checkpoint is a {type(blob).__name__}, not a state dict")
    return blob


# ---------------------------------------------------------------------------
# Checkpoint/config contract guard (2026-08-14).
#
# Every driver instantiates the model from ``cfg.model`` and *then* loads
# weights into it, so the packing order at run time is the YAML's — not the
# one the weights were trained on. The two disagree silently: a
# ``channel_layout`` flip permutes the upper-air block without moving a single
# parameter shape, so ``load_state_dict`` is clean AND the Phase-12h shape
# digest is identical (measured: XDDCWrapper v1 and v2 hash the same). That is
# the exact failure mode this whole rebaseline kept hitting, so it gets a
# guard rather than a paragraph.
#
# ``Module.save`` stores the FULLY RESOLVED constructor kwargs in the
# archive's ``args.json`` (defaults included — verified), so the contract the
# weights were built under is always recoverable.
# ---------------------------------------------------------------------------

#: Constructor kwargs that change how channels are PACKED or interpreted
#: without necessarily changing any parameter shape. A mismatch in any of
#: these is silently wrong, which is why they are checked rather than left to
#: ``load_state_dict``. (Ordering matters as much as membership: a permuted
#: variable list is shape-preserving too.)
_CONTRACT_KEYS = (
    "channel_layout",
    "surface_variables",
    "upper_air_variables",
    "diagnostic_variables",
    "constant_boundary_variables",
    "varying_boundary_variables",
    "scalar_routed_boundary_variables",
    "ocean_state_variables",
    "levels",
)


def mdlus_stored_args(path: str | Path) -> Optional[dict]:
    """The constructor kwargs recorded inside a ``.mdlus`` archive.

    Returns ``None`` when the file is not a readable ``.mdlus`` (a ``.pt``
    state dict, a truncated archive, a future format) — callers treat that as
    "cannot verify" and warn, never as "verified".
    """
    path = Path(path)
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as archive:
                blob = json.loads(archive.read("args.json"))
        elif tarfile.is_tarfile(path):
            with tarfile.open(path, "r") as tar:
                member = tar.extractfile("args.json")
                if member is None:
                    return None
                blob = json.loads(member.read())
        else:
            return None
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    args = blob.get("__args__", blob)
    return args if isinstance(args, dict) else None


def _normalize_contract_element(value: Any) -> Any:
    """One list element, in a form that compares across representations.

    Numbers become floats and everything else becomes its string: the AMIP
    configs write ``levels: [5, 7, 10, ...]`` as ints while the translator's
    auto-derive path takes them off upstream hparams as floats, and ``850 !=
    850.0`` would be a false positive on the one comparison people make most.
    ``bool`` is excluded from the numeric branch because it is an ``int``
    subclass and ``True == 1.0`` would erase a real difference.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


def _normalize_contract_value(value: Any) -> Any:
    """Normalize a whole contract value, element-wise for sequences.

    OmegaConf hands back ``ListConfig`` rather than ``list``, hence the
    duck-typed sequence test rather than an ``isinstance`` against ``list``.
    """
    if isinstance(value, (list, tuple)) or type(value).__name__ == "ListConfig":
        return [_normalize_contract_element(v) for v in value]
    return _normalize_contract_element(value)


def assert_checkpoint_contract(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    *,
    log: Any = None,
    warn_keys: tuple[str, ...] = (),
) -> dict[str, tuple]:
    """Refuse to load weights whose stored channel contract differs.

    Compares :data:`_CONTRACT_KEYS` between the archive's ``args.json`` and
    the instantiated ``model``. Raises :class:`ValueError` listing every
    disagreement; returns the diff either way so a caller can log it.

    ``warn_keys`` downgrades specific keys to a warning. A *warm start* needs
    this: ``training.partial_checkpoint`` exists precisely to load one config's
    weights into a differently-shaped one (the Phase-12f ocean variant adds
    ``ocean_state_variables``), and its own skipped-key report is the intended
    output. ``channel_layout`` is never in ``warn_keys`` at any call site —
    there is no version of "warm start from differently-packed weights" that
    means anything.

    Keys absent from either side are skipped: an artifact predating a kwarg
    cannot be judged against it, and a wrapper family without ocean support has
    nothing to compare. None of this substitutes for ``load_state_dict``'s
    shape check — it catches only what that cannot.
    """
    log = log or logger
    inner = model.module if hasattr(model, "module") else model
    stored = mdlus_stored_args(ckpt_path)
    if stored is None:
        log.warning(
            f"cannot read a channel contract out of {ckpt_path} — skipping the "
            f"contract check. Verify by hand that this checkpoint was trained "
            f"on channel_layout={getattr(inner, 'channel_layout', '?')!r}."
        )
        return {}

    diff: dict[str, tuple] = {}
    for key in _CONTRACT_KEYS:
        if key not in stored or not hasattr(inner, key):
            continue
        want = _normalize_contract_value(stored[key])
        have = _normalize_contract_value(getattr(inner, key))
        if want != have:
            diff[key] = (want, have)

    def _fmt(keys) -> str:
        return "\n".join(
            f"  {key}:\n      checkpoint: {diff[key][0]}\n      cfg.model:  {diff[key][1]}"
            for key in keys
        )

    fatal = [k for k in diff if k not in warn_keys]
    soft = [k for k in diff if k in warn_keys]
    if soft:
        log.warning(
            f"channel contract differs from {Path(ckpt_path).name} in keys this "
            f"call tolerates:\n{_fmt(soft)}"
        )
    if fatal:
        raise ValueError(
            f"channel-contract mismatch between {ckpt_path} and cfg.model "
            f"({type(inner).__name__}):\n"
            + _fmt(fatal)
            + "\n\nThese kwargs change how channels are packed, not how many, so "
            "the load would succeed and the run would be silently wrong. Fix "
            "cfg.model to match the checkpoint (e.g. "
            "'++model.channel_layout=<value>' for a translated v1 artifact), or "
            "load through Module.from_checkpoint, which rebuilds the wrapper "
            "from these stored args."
        )

    if not diff:
        log.info(
            f"channel contract verified against {Path(ckpt_path).name}: "
            f"channel_layout={stored.get('channel_layout', '<absent>')!r}"
        )
    return diff


def find_mdlus_for_model(
    model: torch.nn.Module, ckpt_dir: str | Path
) -> Optional[Path]:
    """The ``.mdlus`` in ``ckpt_dir`` that ``load_checkpoint`` would pick.

    Mirrors :func:`physicsnemo.utils.checkpoint.load_checkpoint`'s lookup:
    files are named ``<ClassName>.<model_parallel_rank>.<index>.mdlus`` and the
    highest index wins. Returns ``None`` when nothing matches, which is not an
    error here — ``load_checkpoint`` logs its own miss.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    inner = model.module if hasattr(model, "module") else model
    name = type(inner).__name__
    best: tuple[int, Optional[Path]] = (-1, None)
    for candidate in ckpt_dir.glob(f"{name}.*.mdlus"):
        parts = candidate.name[len(name) + 1 : -len(".mdlus")].split(".")
        try:
            index = int(parts[-1])
        except (IndexError, ValueError):
            index = 0
        if index > best[0]:
            best = (index, candidate)
    return best[1]


def assert_checkpoint_dir_contract(
    model: torch.nn.Module, ckpt_dir: str | Path, *, log: Any = None
) -> None:
    """:func:`assert_checkpoint_contract` against a checkpoint *directory*."""
    path = find_mdlus_for_model(model, ckpt_dir)
    if path is None:
        return
    assert_checkpoint_contract(model, path, log=log)


#: Warm start deliberately crosses config shapes, so only the packing order is
#: fatal there. See :func:`assert_checkpoint_contract`.
_WARM_START_WARN_KEYS = tuple(k for k in _CONTRACT_KEYS if k != "channel_layout")


def load_partial_weights(
    model: torch.nn.Module,
    partial_ckpt: str | Path,
    *,
    log: Any = None,
) -> dict[str, list[str]]:
    """Warm-start ``model`` from every *compatible* key of a checkpoint.

    Mirrors upstream amip's ``load_partial_weights``: keys present in both
    with matching shapes are loaded, anything else is skipped loudly. Used
    to start an ocean-variant run (Phase 12f) from a no-ocean ERDM
    checkpoint — where the expectation is **zero skipped keys**, because the
    ocean block only *adds* parameters (``input_embed.ocean_embed``, the
    head's ``ocean_experts`` / ``ocean_gate``) and widens nothing. A nonzero
    skip count means the two configs differ somewhere else as well, which is
    worth seeing before burning GPU hours on it — hence the report rather
    than a silent ``strict=False`` load.

    Returns ``{"loaded": [...], "skipped": [...], "fresh": [...]}`` —
    ``fresh`` being the target's newly initialised keys.
    """
    log = log or logger
    path = Path(partial_ckpt)
    if not path.exists():
        raise FileNotFoundError(f"partial_checkpoint {path} does not exist")

    if path.suffix == ".mdlus":
        # Shape-compatible is not contract-compatible: fork-packed weights fit
        # a v2 wrapper's tensors exactly and would poison every channel.
        assert_checkpoint_contract(
            model, path, log=log, warn_keys=_WARM_START_WARN_KEYS
        )
        src = _mdlus_state_dict(path)
    else:
        src = _extract_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)
        )
    src = {_strip_wrap_prefixes(k): v for k, v in src.items()}

    target = model.state_dict()
    filtered, skipped = {}, []
    for k, v in src.items():
        if k in target and getattr(v, "shape", None) == target[k].shape:
            filtered[k] = v
        else:
            reason = (
                "not in model"
                if k not in target
                else f"shape {tuple(v.shape)} != {tuple(target[k].shape)}"
            )
            skipped.append(f"{k} ({reason})")

    model.load_state_dict(filtered, strict=False)
    fresh = [k for k in target if k not in filtered]

    # f-strings, not %-args: the recipes pass physicsnemo's ``PythonLogger``,
    # whose info()/warning() take a single message argument.
    log.info(
        f"partial checkpoint {path}: loaded {len(filtered)}/{len(src)} keys, "
        f"{len(fresh)} left at init"
    )
    if skipped:
        log.warning(
            f"partial checkpoint: SKIPPED {len(skipped)} key(s) — the source "
            f"and this config disagree beyond added parameters:"
        )
        for k in skipped:
            log.warning(f"  {k}")
    if fresh:
        log.info(
            f"partial checkpoint: {len(fresh)} key(s) newly initialised "
            f"(e.g. {', '.join(fresh[:5])})"
        )
    return {"loaded": sorted(filtered), "skipped": skipped, "fresh": fresh}


def repair_incomplete_slurm_env(*, log=None) -> bool:
    """Make a bare ``python train_diffusion.py`` under ``sbatch`` launch correctly.

    ``DistributedManager.initialize()`` tries the ENV method, falls through to the
    SLURM branch on TypeError, and there reads ``SLURM_LAUNCH_NODE_IPADDR`` as its
    address. Measured 2026-08-18: an **sbatch** shell exports ``SLURM_PROCID=0`` and
    ``SLURM_NPROCS=1`` but NOT that IP — only ``srun`` steps get it — so the manager
    hands ``addr=None`` to ``setup()`` and the run dies two frames later with::

        TypeError: str expected, not NoneType     (os.environ["MASTER_ADDR"] = addr)

    Nothing in that message names SLURM, the missing variable, or the launcher, and
    it fires before the model or dataset is touched, which makes it read like a
    config error. It cost two debugging rounds in one day — once in a job script of
    mine, once in an ad-hoc benchmark script that reintroduced it after the first fix
    — so the recipe should not depend on the caller remembering to use torchrun.

    The narrow repair: when SLURM says single-process (``SLURM_NPROCS`` unset or 1),
    the launch IP is absent, and no ``RANK`` is set, fill in the ENV-method variables
    for one rank so ``initialize_env()`` succeeds on its own terms. Under ``srun``
    the IP exists and this does nothing; under ``torchrun`` ``RANK`` exists and this
    does nothing; genuinely multi-rank SLURM launches are untouched.
    """
    if os.environ.get("RANK") is not None:
        return False                      # torchrun (or an explicit ENV launch)
    if os.environ.get("SLURM_PROCID") is None:
        return False                      # not under SLURM at all
    if os.environ.get("SLURM_LAUNCH_NODE_IPADDR") is not None:
        return False                      # an srun step: the SLURM branch works
    if int(os.environ.get("SLURM_NPROCS", "1") or 1) > 1:
        return False                      # multi-rank: do not paper over it
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    if log is not None:
        log.info(
            "single-process SLURM launch without srun: filled RANK/WORLD_SIZE/"
            "MASTER_ADDR so the ENV init path applies (SLURM_LAUNCH_NODE_IPADDR "
            "is unset in an sbatch shell, which would otherwise fail with "
            "'str expected, not NoneType')"
        )
    return True


_MATMUL_PRECISIONS = ("highest", "high", "medium")


def apply_math_precision(cfg_train: Any, *, log=None) -> dict:
    """Opt-in float32 math knobs. Default: change nothing.

    Measured on an A100-40GB with the v2 x_DDC config at its shipped geometry
    (``benchmarks/.../bench_v2_training_step.py``, 2026-08-18): one training step
    takes **656 ms** with torch's default ``highest`` matmul precision and
    **365 ms** with ``high`` — a 1.80x speedup for a mantissa change that stays
    within ~3 decimals of fp32, well under training noise.

    This is a PARITY gap, not a bonus: upstream amip_v2's ``train.py`` calls
    ``torch.set_float32_matmul_precision("high")``, so its runs already get the
    365 ms while ours got 656. ``train.py`` in this repo does the same for
    Pangu/SFNO and records ~15% there. ``train_diffusion.py`` never did.

    Left OFF by default all the same, because it changes the numerics of a
    trained model and that is the user's call, not this function's. Enable with::

        ++training.matmul_precision=high        # the 1.80x
        ++training.allow_tf32=true              # cuDNN convolutions too
        ++training.cudnn_benchmark=true         # autotune conv algos
        ++training.disable_cudnn_sdpa=true      # REQUIRED for bf16 on GH200
        ++training.attention_dtype=bf16         # attention only; see the profile

    Returns what it applied, so a run's log and any benchmark record agree about
    the settings the numbers were produced under.
    """
    applied: dict[str, Any] = {}
    if cfg_train is None:
        return applied
    precision = cfg_train.get("matmul_precision", None)
    tf32 = cfg_train.get("allow_tf32", None)
    benchmark = cfg_train.get("cudnn_benchmark", None)
    no_cudnn_sdpa = cfg_train.get("disable_cudnn_sdpa", None)
    attn_dtype = cfg_train.get("attention_dtype", None)

    if precision is not None:
        if str(precision) not in _MATMUL_PRECISIONS:
            raise ValueError(
                f"training.matmul_precision={precision!r} is not one of "
                f"{_MATMUL_PRECISIONS}"
            )
        torch.set_float32_matmul_precision(str(precision))
        applied["matmul_precision"] = str(precision)
    if tf32 is not None and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        applied["allow_tf32"] = bool(tf32)
    if benchmark is not None:
        torch.backends.cudnn.benchmark = bool(benchmark)
        applied["cudnn_benchmark"] = bool(benchmark)
    if no_cudnn_sdpa:
        # GH200 escape hatch (measured 2026-08-18). With DeltaAI's inherited
        # torch 2.10, bf16 attention on GH200 raises "cuDNN Frontend error: No
        # valid execution plans built" out of F.scaled_dot_product_attention
        # instead of falling back, so `amp: bf16` cannot run there at all.
        # Disabling that one backend leaves flash and mem-efficient, which work
        # -- and bf16 is the largest lever measured: 3.83x on ERDM v2 and 3.89x
        # on x_DDC versus fp32, with a third off ERDM's peak memory.
        # A no-op on x86, where the cuDNN backend is fine.
        torch.backends.cuda.enable_cudnn_sdp(False)
        applied["disable_cudnn_sdpa"] = True
    if attn_dtype is not None:
        # Profile-driven (Nsight Systems on GH200, 2026-08-18): with TF32 on,
        # ~70% of a training step is fp32 mem-efficient attention running sm80
        # CUTLASS kernels, which the TF32 flag cannot touch — ERDM's attention
        # backward measured 3488.8 ms fp32 vs 3510.7 ms TF32, i.e. unchanged,
        # while its GEMMs fell from 6416 ms to 705 ms. Running ONLY attention in
        # bf16 reaches the sm90 flash kernels with everything else left fp32.
        from physicsnemo.experimental.models.amip_si import set_attention_dtype

        applied["attention_dtype"] = str(set_attention_dtype(attn_dtype))

    # Messages are pre-formatted, NOT passed as %-style varargs: the recipe hands
    # in PhysicsNeMo's PythonLogger, whose info() takes a single string
    # (physicsnemo/utils/logging/console.py). Passing args raised
    # "PythonLogger.info() takes 2 positional arguments but 3 were given" and, since
    # the no-knobs branch logs too, broke EVERY diffusion run — caught by
    # test_train_diffusion_smoke, which drives the real entry point.
    if log is not None:
        precision_now = torch.get_float32_matmul_precision()
        if applied:
            log.info(
                f"float32 math knobs applied: {applied} "
                f"(effective matmul precision {precision_now})"
            )
        else:
            log.info(
                f"float32 math knobs: none set — matmul precision {precision_now}. "
                f"++training.matmul_precision=high measured 1.80x on x_DDC/A100 "
                f"and is what upstream amip_v2 trains with."
            )
    return applied


def make_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build an optimizer from a config dict-like.

    Recognized keys:

    * ``optimizer_type`` — ``"AdamW"`` or ``"Muon"``.
    * ``lr`` — base learning rate.
    * ``weight_decay`` — default 0.
    * ``fused`` — when True, requests the fused CUDA kernel for AdamW
      (``torch.optim.AdamW(..., fused=True)``). Requires CUDA; falls back to
      the eager AdamW with a warning if the runtime can't honor it. Defaults
      to True on CUDA (matches PanguWeather's reference SFNO trainer), False
      otherwise.

    ``optimizer_type="Muon"`` requires ``model`` to expose a
    ``muon_param_groups(lr, weight_decay)`` method (the amip_si wrappers —
    :class:`AmipDiTWrapper` / :class:`RollingDiTWrapper` / :class:`ERDMWrapper`
    — all do) and the ``Muon`` package
    (``pip install git+https://github.com/KellerJordan/Muon``, or the
    ``muon-optimizers`` extra in ``pyproject.toml``). The two param groups it
    returns are handed verbatim to ``muon.MuonWithAuxAdam``.
    """
    name = getattr(cfg, "optimizer_type", "AdamW")
    if name == "Muon":
        return _make_muon_optimizer(model, cfg)
    if name != "AdamW":
        raise ValueError(
            f"Unsupported optimizer_type={name!r} (supported: 'AdamW', 'Muon')."
        )
    fused = bool(getattr(cfg, "fused", torch.cuda.is_available()))
    wd = float(getattr(cfg, "weight_decay", 0.0))
    betas = getattr(cfg, "betas", None)
    kwargs = dict(lr=float(cfg.lr), weight_decay=wd)
    if betas is not None:
        kwargs["betas"] = tuple(float(b) for b in betas)
    if fused:
        if not torch.cuda.is_available():
            import warnings as _warnings

            _warnings.warn(
                "cfg.fused=True requested but CUDA is not available; falling "
                "back to eager AdamW.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            kwargs["fused"] = True

    # Selective weight decay (ArchesWeather): apply wd ONLY to params whose name
    # contains 'weight' and not 'norm' (i.e. Linear/Conv weights, not biases or
    # LayerNorm/pos-bias params). Matches geoarches' configure_optimizers.
    if bool(getattr(cfg, "selective_weight_decay", False)) and wd > 0.0:
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "weight" in name and "norm" not in name:
                decay.append(p)
            else:
                no_decay.append(p)
        params = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs.pop("weight_decay")
        return torch.optim.AdamW(params, **kwargs)

    return torch.optim.AdamW(model.parameters(), **kwargs)


def _make_muon_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build ``muon.MuonWithAuxAdam`` from ``model.muon_param_groups()``.

    ``cfg.weight_decay`` is forwarded to both the Muon and aux-AdamW
    groups (matches upstream amip, which applies a single weight_decay
    across both). The Muon group's LR multiplier defaults to the
    wrapper method's own default (10x, per upstream).
    """
    if not hasattr(model, "muon_param_groups"):
        raise ValueError(
            f"optimizer_type='Muon' requires a model exposing "
            f"muon_param_groups(); {type(model).__name__} does not. "
            "Use one of the amip_si wrappers (AmipDiTWrapper / "
            "RollingDiTWrapper / ERDMWrapper) or add Muon support to "
            "the wrapper."
        )
    try:
        from muon import MuonWithAuxAdam
    except ImportError as exc:
        raise ImportError(
            "optimizer_type='Muon' requires the `muon` package: "
            "`pip install git+https://github.com/KellerJordan/Muon` "
            "(or `pip install nvidia-physicsnemo[muon-optimizers]`)."
        ) from exc
    # MuonWithAuxAdam.step() calls dist.get_world_size() unconditionally — it
    # pads its parameter list to a multiple of the world size — so it cannot
    # run outside a process group at all. Without this check the failure lands
    # mid-training on the FIRST optimizer step, as a ValueError from
    # distributed_c10d with no mention of Muon or of how to launch (hit on
    # Polaris smoke job 7438576).
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "optimizer_type='Muon' needs an initialized torch.distributed "
            "process group (MuonWithAuxAdam.step() shards its parameter list "
            "across the world size). Launch under torchrun — "
            "`torchrun --standalone --nproc-per-node=1 <script>` is enough for "
            "a single-process run — or set optimizer.type=AdamW."
        )
    param_groups = model.muon_param_groups(
        lr=float(cfg.lr),
        weight_decay=float(getattr(cfg, "weight_decay", 0.01)),
    )
    return MuonWithAuxAdam(param_groups)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a scheduler from a config dict-like.

    Supported ``scheduler``:

    * ``"OneCycleLR"`` — uses ``oc_pct_start``, ``oc_div_factor``,
      ``oc_final_div_factor`` (PanguWeather PANGU_PLASIM_H5_DERECHO_0514 keys).
      ``max_lr`` defaults to ``lr``.
    * ``"LinearWarmupCosineAnnealingLR"`` — composes a linear warmup
      (``num_warmup_steps``) with cosine annealing to ``eta_min``.
    """
    name = getattr(cfg, "scheduler", "OneCycleLR")
    if name == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=float(cfg.lr),
            total_steps=total_steps,
            pct_start=float(getattr(cfg, "oc_pct_start", 0.1)),
            div_factor=float(getattr(cfg, "oc_div_factor", 1e5)),
            final_div_factor=float(getattr(cfg, "oc_final_div_factor", 0.00025)),
            anneal_strategy="cos",
        )
    if name == "LinearWarmupCosineAnnealingLR":
        warmup_steps = int(getattr(cfg, "num_warmup_steps", 0) or 0)
        warmup_start_lr = float(getattr(cfg, "warmup_start_lr", 1e-8))
        eta_min = float(getattr(cfg, "eta_min", 0.0))
        if warmup_steps <= 0:
            return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
        warmup = LinearLR(
            optimizer,
            start_factor=warmup_start_lr / float(cfg.lr),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    if name == "CosineAnnealingLR":
        # Plain CosineAnnealingLR — used by the AMIP diffusion recipe.
        # ``T_max`` defaults to ``total_steps`` (the per-stage budget the
        # caller supplies); ``cosine_eta_min`` mirrors the yaml key name
        # used in conf/training/amip_diffusion.yaml. ``eta_min`` is
        # accepted as a synonym so the Phase 3 LinearWarmupCosineAnnealingLR
        # config keys also work here.
        T_max = int(getattr(cfg, "T_max", total_steps))
        eta_min = float(
            getattr(cfg, "cosine_eta_min", getattr(cfg, "eta_min", 0.0))
        )
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    raise ValueError(f"Unknown scheduler {name!r}")


_AMP_DTYPES = {
    "none": None,
    "off": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _resolve_amp_dtype(amp: str | bool | None) -> Optional[torch.dtype]:
    """Map ``cfg.amp`` (string or bool) to a torch dtype or ``None`` for off."""
    if amp is None or amp is False:
        return None
    if amp is True:
        return torch.bfloat16  # default-on AMP picks bf16 (matches PanguWeather)
    return _AMP_DTYPES.get(str(amp).lower())


def train_step(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    batch: dict[str, torch.Tensor],
    has_diagnostic: bool,
    vae_kl_weight: float = 0.0,
    amp_dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional["torch.amp.GradScaler"] = None,
) -> dict[str, torch.Tensor]:
    """One optimizer step: forward + backward + step + scheduler tick.

    Returns the loss dict from :class:`PanguPlasimLoss` plus a ``"vae_kl"``
    entry. Compatible with both PanguPlasimLegacy (5- or 7-tuple output with
    zero latent placeholders — `vae_kl` stays ~0) and PanguPlasim with VAE
    (6- or 7-tuple with real ``mu``/``logvar``).

    Mixed-precision support
    -----------------------
    When ``amp_dtype`` is not ``None``, the forward + loss computation runs
    under ``torch.amp.autocast(device_type="cuda", dtype=amp_dtype)``. For
    ``bf16`` (matches PanguWeather v2.0's default for SFNO_PLASIM) no
    :class:`GradScaler` is needed. For ``fp16`` pass an externally-managed
    ``grad_scaler`` so the trainer can also persist its state across
    checkpoints. The optimizer step is wrapped in
    ``grad_scaler.step`` + ``grad_scaler.update`` when present.

    When ``vae_kl_weight > 0`` and the model emits real ``(mu, logvar,
    mu_e2, logvar_e2)`` tuples, the KL divergence between the two encoder
    posteriors is computed and added: ``total = task_loss + vae_kl_weight * kl``.
    For PanguPlasimLegacy the model returns zero placeholders for the latent
    fields, so the KL evaluates to 0 and the task loss is unchanged
    regardless of ``vae_kl_weight``.
    """
    from loss import vae_kl_loss  # local import keeps train_loop / loss decoupled at import time

    optimizer.zero_grad(set_to_none=True)

    # Autocast context — no-op when amp_dtype is None.
    if amp_dtype is None:
        amp_ctx = contextlib.nullcontext()
    else:
        device_type = "cuda" if batch["surface_in"].is_cuda else "cpu"
        amp_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)

    extra_kwargs = _optional_model_kwargs(model, batch)
    with amp_ctx:
        out = model(
            batch["surface_in"],
            batch["constant_boundary"],
            batch["varying_boundary"],
            batch["upper_air_in"],
            target_surface=batch.get("target_surface"),
            target_upper_air=batch.get("target_upper_air"),
            train=True,
            **extra_kwargs,
        ) if _model_accepts_train_kwarg(model) else model(
            batch["surface_in"],
            batch["constant_boundary"],
            batch["varying_boundary"],
            batch["upper_air_in"],
            **extra_kwargs,
        )

        # Output tuple layout:
        # * PanguPlasimLegacy (no diag): (surface, upper_air, 0, 0, 0, 0)
        # * PanguPlasimLegacy (diag):    (surface, upper_air, diag, 0, 0, 0, 0)
        # * PanguPlasim (no diag, train=True): (surface, upper_air, mu, logvar, mu_e2, logvar_e2)
        # * PanguPlasim (diag, train=True):    (surface, upper_air, diag, mu, logvar, mu_e2, logvar_e2)
        if has_diagnostic:
            out_surface, out_upper_air, out_diag = out[0], out[1], out[2]
            latent_offset = 3
        else:
            out_surface, out_upper_air = out[0], out[1]
            out_diag = None
            latent_offset = 2

        losses = loss_fn(
            out_surface,
            out_upper_air,
            batch["target_surface"],
            batch["target_upper_air"],
            out_diagnostic=out_diag,
            target_diagnostic=batch.get("diagnostic") if has_diagnostic else None,
        )

        # The VAE-KL branch fires only when (a) KL weight > 0, (b) the model
        # returned at least four latent slots, AND (c) those slots are torch
        # Tensors (the legacy port emits Python int `0` placeholders, not
        # tensors — easy sentinel for "no VAE here").
        latent_slots = out[latent_offset : latent_offset + 4] if len(out) >= latent_offset + 4 else ()
        has_real_latents = (
            len(latent_slots) == 4
            and all(isinstance(x, torch.Tensor) and x.numel() > 0 for x in latent_slots)
        )
        if vae_kl_weight > 0.0 and has_real_latents:
            mu, logvar, mu_e2, logvar_e2 = latent_slots
            kl = vae_kl_loss(mu, logvar, mu_e2, logvar_e2)
            losses["vae_kl"] = kl.detach()
            losses["loss"] = losses["loss"] + vae_kl_weight * kl
        else:
            # VAE disabled or model emits placeholders. Keep the key for logger uniformity.
            losses["vae_kl"] = torch.zeros((), device=out_surface.device, dtype=out_surface.dtype)

    # Backward + step. GradScaler is required for fp16 (underflow protection);
    # bf16 retains enough dynamic range that no scaling is needed.
    if grad_scaler is not None:
        grad_scaler.scale(losses["loss"]).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        losses["loss"].backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return losses


def multistep_train_step(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    batch: dict[str, torch.Tensor],
    has_diagnostic: bool,
    unroll_steps: int,
    vae_kl_weight: float = 0.0,
    amp_dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional["torch.amp.GradScaler"] = None,
) -> dict[str, torch.Tensor]:
    r"""K-step rollout training with per-step loss accumulation.

    Expects ``batch`` to carry sequence keys produced by
    :class:`physicsnemo.experimental.datapipes.plasim.SequenceDataset`:

    * ``surface_in_seq``:        ``(B, T+1, C_s, H, W)``
    * ``upper_air_in_seq``:      ``(B, T+1, C_u, L, H, W)``
    * ``varying_boundary_seq``:  ``(B, T+1, C_b, H, W)``
    * ``diagnostic_seq``:        ``(B, T+1, C_d, H, W)`` (when has_diagnostic)
    * ``constant_boundary``:     ``(C_b^c, H, W)`` or ``(B, C_b^c, H, W)``

    The model is unrolled K times (K = ``unroll_steps``); the prediction
    at step k is fed back as the input state for step k+1. Per-step
    losses are summed then divided by K — the resulting scalar is in the
    same scale as the single-step loss for direct LR/EMA comparability.

    VAE-KL is not supported in this code path (the multi-step rollout
    averages predictions away from the latent encoder semantics); pass
    ``vae_kl_weight=0`` (default) or use single-step
    :func:`train_step` for the VAE variant.
    """
    if "surface_in_seq" not in batch or "upper_air_in_seq" not in batch:
        raise KeyError(
            "multistep_train_step requires sequence batch keys "
            "(`*_seq`). Use the datapipe in unroll_steps>1 mode."
        )
    if int(unroll_steps) < 1:
        raise ValueError(f"unroll_steps must be ≥ 1, got {unroll_steps}")

    optimizer.zero_grad(set_to_none=True)

    if amp_dtype is None:
        amp_ctx = contextlib.nullcontext()
    else:
        device_type = "cuda" if batch["surface_in_seq"].is_cuda else "cpu"
        amp_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)

    surface_seq = batch["surface_in_seq"]               # (B, T+1, C_s, H, W)
    upper_seq = batch["upper_air_in_seq"]               # (B, T+1, C_u, L, H, W)
    varying_seq = batch["varying_boundary_seq"]         # (B, T+1, C_b, H, W)
    diag_seq = batch.get("diagnostic_seq") if has_diagnostic else None
    const_boundary = batch.get("constant_boundary")     # (C, H, W) or (B, C, H, W)

    # Initial state = first frame.
    state_surface = surface_seq[:, 0]
    state_upper = upper_seq[:, 0]

    accum_components = {
        "surface": torch.zeros((), device=state_surface.device, dtype=state_surface.dtype),
        "upper_air": torch.zeros((), device=state_surface.device, dtype=state_surface.dtype),
        "diagnostic": torch.zeros((), device=state_surface.device, dtype=state_surface.dtype),
    }
    accum_loss = torch.zeros((), device=state_surface.device, dtype=state_surface.dtype)

    with amp_ctx:
        for k in range(int(unroll_steps)):
            boundary_in = varying_seq[:, k]
            out = model(
                state_surface,
                const_boundary,
                boundary_in,
                state_upper,
            )
            if has_diagnostic:
                next_surface, next_upper, next_diag = out[0], out[1], out[2]
            else:
                next_surface, next_upper = out[0], out[1]
                next_diag = None

            target_surface_k = surface_seq[:, k + 1]
            target_upper_k = upper_seq[:, k + 1]
            target_diag_k = diag_seq[:, k + 1] if diag_seq is not None else None

            losses_k = loss_fn(
                next_surface,
                next_upper,
                target_surface_k,
                target_upper_k,
                out_diagnostic=next_diag,
                target_diagnostic=target_diag_k,
            )
            accum_loss = accum_loss + losses_k["loss"]
            for comp in ("surface", "upper_air", "diagnostic"):
                if comp in losses_k:
                    val = losses_k[comp]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor(float(val), device=accum_loss.device, dtype=accum_loss.dtype)
                    accum_components[comp] = accum_components[comp] + val

            # Detach the boundary path (no grad through it) but keep the
            # state path so per-step gradients flow back through the rollout.
            state_surface = next_surface
            state_upper = next_upper

    total = accum_loss / float(unroll_steps)
    avg_components = {
        k: (v / float(unroll_steps)).detach() for k, v in accum_components.items()
    }
    losses_out: dict[str, torch.Tensor] = {
        "loss": total,
        "surface": avg_components["surface"],
        "upper_air": avg_components["upper_air"],
        "diagnostic": avg_components["diagnostic"],
        "vae_kl": torch.zeros((), device=total.device, dtype=total.dtype),
    }

    if grad_scaler is not None:
        grad_scaler.scale(total).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        total.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return losses_out


_OPTIONAL_MODEL_BATCH_KEYS = ("surface_prev_in", "upper_air_prev_in", "calendar")


def _optional_model_kwargs(
    model: torch.nn.Module, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Extra forward kwargs a model wants that also exist in the batch.

    Returns the subset of ``{surface_prev_in, upper_air_prev_in, calendar}``
    that are BOTH present in ``batch`` AND named parameters of the model's
    ``forward``. SFNO/Pangu forwards don't name these, so the result is empty
    and their call is byte-for-byte unchanged; ArchesWeather names all three.
    """
    inner = model.module if hasattr(model, "module") else model
    varnames = getattr(
        inner.forward, "__code__", type("_x", (), {"co_varnames": ()})()
    ).co_varnames
    return {k: batch[k] for k in _OPTIONAL_MODEL_BATCH_KEYS if k in batch and k in varnames}


def _model_accepts_train_kwarg(model: torch.nn.Module) -> bool:
    """Detect whether the model's forward signature accepts ``train=`` + targets.

    The faithful PanguPlasim port takes a ``train`` flag plus optional
    ``target_*`` kwargs (it routes them through the VAE's second encoder when
    ``train=True``). PanguPlasimLegacy doesn't — its forward only takes the
    four input tensors.
    """
    inner = model.module if hasattr(model, "module") else model
    return getattr(inner, "has_vae", False) or "train" in getattr(
        inner.forward, "__code__", type("_x", (), {"co_varnames": ()})()
    ).co_varnames

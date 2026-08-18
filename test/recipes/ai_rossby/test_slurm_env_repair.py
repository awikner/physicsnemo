# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""A bare `python train_diffusion.py` under sbatch must launch (2026-08-18).

``DistributedManager.initialize()`` tries the ENV method, falls through to the SLURM
branch on ``TypeError``, and reads ``SLURM_LAUNCH_NODE_IPADDR`` as its address.
Measured: an **sbatch** shell exports ``SLURM_PROCID=0`` and ``SLURM_NPROCS=1`` but
NOT that IP — only ``srun`` steps get it — so ``setup()`` receives ``addr=None`` and
the run dies with ``TypeError: str expected, not NoneType`` from
``os.environ["MASTER_ADDR"] = addr``.

That message names neither SLURM, nor the missing variable, nor the launcher, and it
fires before the model or dataset is built, so it reads like a config error. It cost
two debugging rounds in one day: once in a job script, then again in an ad-hoc
benchmark script that reintroduced it after the first fix. Hence a repair in the
recipe rather than a rule the caller has to remember.

The repair must be NARROW — these tests pin each way it declines to act, because
papering over a genuine multi-rank launch would turn a 4-GPU run into four
one-GPU runs that all think they are rank 0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_RECIPE = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
if str(_RECIPE) not in sys.path:
    sys.path.insert(0, str(_RECIPE))

from train_loop import repair_incomplete_slurm_env  # noqa: E402

_TOUCHED = (
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
    "SLURM_PROCID", "SLURM_NPROCS", "SLURM_LOCALID", "SLURM_LAUNCH_NODE_IPADDR",
)


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k) for k in _TOUCHED}
    for k in _TOUCHED:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_the_sbatch_shell_case_is_repaired():
    """SLURM_PROCID set, no launch IP, one rank — the case that failed."""
    os.environ["SLURM_PROCID"] = "0"
    os.environ["SLURM_NPROCS"] = "1"
    os.environ["SLURM_LOCALID"] = "0"
    assert repair_incomplete_slurm_env() is True
    assert os.environ["RANK"] == "0"
    assert os.environ["WORLD_SIZE"] == "1"
    assert os.environ["MASTER_ADDR"]          # the None that broke setup()
    assert os.environ["MASTER_PORT"]


def test_an_srun_step_is_left_alone():
    """srun provides the launch IP, so the manager's SLURM branch works as designed."""
    os.environ["SLURM_PROCID"] = "0"
    os.environ["SLURM_NPROCS"] = "1"
    os.environ["SLURM_LAUNCH_NODE_IPADDR"] = "141.142.253.158"
    assert repair_incomplete_slurm_env() is False
    assert "RANK" not in os.environ


def test_a_torchrun_launch_is_left_alone():
    """RANK present means the ENV path already applies; touching it would fight
    torchrun's own rendezvous."""
    os.environ.update(RANK="3", WORLD_SIZE="4", MASTER_ADDR="10.0.0.1",
                      MASTER_PORT="29500", SLURM_PROCID="3")
    assert repair_incomplete_slurm_env() is False
    assert os.environ["RANK"] == "3"
    assert os.environ["WORLD_SIZE"] == "4"


def test_a_genuine_multirank_slurm_launch_is_refused():
    """The important refusal: filling WORLD_SIZE=1 for a 4-rank job would give four
    processes that each believe they are the only rank."""
    os.environ["SLURM_PROCID"] = "2"
    os.environ["SLURM_NPROCS"] = "4"
    assert repair_incomplete_slurm_env() is False
    assert "WORLD_SIZE" not in os.environ


def test_outside_slurm_nothing_happens():
    assert repair_incomplete_slurm_env() is False
    assert "RANK" not in os.environ


def test_existing_values_are_not_overwritten():
    """setdefault, not assignment: a caller that set MASTER_PORT to dodge a clash
    must keep it."""
    os.environ["SLURM_PROCID"] = "0"
    os.environ["MASTER_PORT"] = "12345"
    assert repair_incomplete_slurm_env() is True
    assert os.environ["MASTER_PORT"] == "12345"


def test_it_logs_through_a_single_argument_logger():
    """The recipe's logger is PhysicsNeMo's PythonLogger — one string, no varargs."""
    messages = []

    class _Log:
        def info(self, message: str):
            if not isinstance(message, str):
                raise TypeError(type(message))
            messages.append(message)

    os.environ["SLURM_PROCID"] = "0"
    assert repair_incomplete_slurm_env(log=_Log()) is True
    assert len(messages) == 1
    assert "srun" in messages[0]


# ---------------------------------------------------------------------------
# DataLoader start method (2026-08-18)
# ---------------------------------------------------------------------------
from train_loop import choose_worker_start_method  # noqa: E402


class _Log:
    def __init__(self):
        self.messages: list[str] = []

    def info(self, message: str):
        if not isinstance(message, str):
            raise TypeError(f"expected a pre-formatted str, got {type(message)}")
        self.messages.append(message)


@pytest.fixture
def _omp():
    saved = os.environ.get("OMP_NUM_THREADS")
    yield
    if saved is None:
        os.environ.pop("OMP_NUM_THREADS", None)
    else:
        os.environ["OMP_NUM_THREADS"] = saved


def test_no_workers_means_no_start_method(_omp):
    os.environ["OMP_NUM_THREADS"] = "8"
    assert choose_worker_start_method(0) is None


def test_one_omp_thread_keeps_the_faster_fork(_omp):
    """Measured safe: OMP_NUM_THREADS=1 with forked workers ran 5/5 batches, and
    every shipped HPC script exports exactly that."""
    os.environ["OMP_NUM_THREADS"] = "1"
    assert choose_worker_start_method(4) is None


@pytest.mark.parametrize("omp", ["2", "8", "16"])
def test_multiple_omp_threads_switch_to_forkserver(_omp, omp):
    """Measured deadlock: OMP_NUM_THREADS=8 with forked workers produced 0 batches
    in 300 s, twice, wedged in the boundary smoothing's conv2d."""
    os.environ["OMP_NUM_THREADS"] = omp
    assert choose_worker_start_method(4) == "forkserver"


def test_an_unset_thread_count_is_treated_as_unsafe(_omp):
    """torch defaults its intra-op pool to the core count when the variable is
    unset, so 'unset' is the multi-threaded case, not the safe one."""
    os.environ.pop("OMP_NUM_THREADS", None)
    assert choose_worker_start_method(4) == "forkserver"


def test_an_explicit_request_wins_over_the_policy(_omp):
    os.environ["OMP_NUM_THREADS"] = "1"
    assert choose_worker_start_method(4, "spawn") == "spawn"


def test_it_says_which_method_it_chose_and_why(_omp):
    """A silent switch would make a throughput change look like a mystery."""
    os.environ["OMP_NUM_THREADS"] = "8"
    log = _Log()
    choose_worker_start_method(4, None, log=log)
    assert "forkserver" in log.messages[0]
    assert "OMP_NUM_THREADS=8" in log.messages[0]

---
name: dsi-smoke-test
description: Submit a pytest target as a GPU smoke test on the UChicago DSI cluster under account general_group, QoS interactive (SLURM, --gres=gpu). Use whenever the user asks to run a smoke test, run GPU tests, or verify a ported physicsnemo feature on DSI. Streams output back; blocks until pytest exits.
---

# dsi-smoke-test

Runs a pytest smoke target on a UChicago DSI GPU node under account `general_group`, QoS
`interactive`, in the ai-rossby workflow defined by `hpc/dsi.md`. SLURM with `--gres=gpu`; the
job streams output and ends when pytest exits. Pairs with `dsi-shell`.

> **⚠️ SKELETON.** Storage paths and the compute-node CUDA driver version are **TBD until
> verified on DSI** (`hpc/dsi.md`). The QoS/`--gres` syntax below is from DSI policy and expected
> correct.

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--account` | `general_group` | Only deviation: user names another account |
| `--qos` | `interactive` | 4 h, non-preemptable; the smoke tier. `general`/`protected` for other work |
| `--time` | `00:30:00` | Bump to `01:00:00` for DDP; keep ≤ the 4 h interactive cap |
| `--nodes` | `1` | Single-node by contract |
| `--ntasks` | `1` | Pytest runs once; DDP driven by `torchrun` inside |
| `--cpus-per-task` | `8` | Dataloader workers |
| `--gres` | `gpu:1` | `gpu:2` for DDP; `gpu:H100:1`/`gpu:A100:1` to pin a type |
| `--mem` | `64G` | |
| `--job-name` | `pn-smoke` | |

## What this skill does, step by step

1. **Confirm the target.** A pytest path, `-k` expression, or node-id. If missing, ask.
2. **Choose GPU count / type.** Default 1 any-type GPU. For "DDP"/"multi-GPU" set `--gres=gpu:2`
   and `--time=01:00:00`. If the user names a GPU type (H100, A100, …), use `--gres=gpu:<type>:N`.
   Never exceed the interactive cap or single node.
3. **Build the command** (no module load — DSI has no module system):
   ```
   cd /net/projects/general_group/awikner/physicsnemo && \
   source .venv/bin/activate && \
   pytest -m "smoke and cuda" -x -q <TARGET>
   ```
   For DDP, prepend `torchrun --standalone --nproc-per-node=2 -m ` before `pytest`.
4. **Submit via `srun`** (stream; job ends when pytest does). Do NOT use `--pty` (that's
   `dsi-shell`) and do NOT queue-and-forget.
5. **Report.** On pass, state the test name + wall time. On failure, show the pytest tail and
   stop — do not silently rerun (and note that `interactive` preempts `general`, so a genuine
   preemption is distinct from a test failure).

## Example (Bash tool)

```bash
srun --account=general_group --qos=interactive --time=00:30:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 \
  --job-name=pn-smoke \
  bash -lc 'cd /net/projects/general_group/awikner/physicsnemo && \
            source .venv/bin/activate && \
            pytest -m "smoke and cuda" -x -q test/models/pangu_plasim/'
```

## Refuse / push back when

- User asks for `--time` beyond the 4 h interactive cap, > the node's GPU count, or multi-node —
  stop and surface it (for longer runs suggest `--qos=general`, 12 h, in a separate script).
- User names an account other than `general_group` without explanation — confirm first.
- The repo is not on `ai-rossby` — note it (smoke tests likely don't exist on `main`).

## Out-of-scope

- Long-running fidelity tests — `--qos=general`/`protected` in a separate job script.
- Multi-node DDP — out of scope for smoke.
- Interactive debugging — use `dsi-shell`.
- Data transfers in/out of DSI — need Globus (Phase 10).
- Any other cluster — its own skill + doc.

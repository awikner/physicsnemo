# Development history & plans (internal)

These are the internal working documents behind the ai-rossby fork — design
notes, phased porting plans, and status logs. They are **development history**,
not user-facing guides, and some contain stale "in progress" status and
collaborator-specific paths. For onboarding and usage, use the top-level
[`README.md`](../../README.md) and the recipe docs under
`examples/weather/ai_rossby/` instead.

| Doc | What it is |
|---|---|
| [`project_outline.md`](project_outline.md) | The fork's overall mission — a unified weather/climate emulator codebase on PhysicsNeMo. |
| [`implementation_plan.md`](implementation_plan.md) | The master phased porting plan (Phases 1–10): PanguWeather & amip → PhysicsNeMo. |
| [`pangu_plasim_reuse_plan.md`](pangu_plasim_reuse_plan.md) | Component-reuse rationale — swapping vendored Pangu blocks for `physicsnemo.nn`. |
| [`phase9_implementation_plan.md`](phase9_implementation_plan.md) | Multi-cluster dev setup (six HPC clusters): SSH, install, env propagation, Nsight/CUDA. |
| [`phase10_implementation_plan.md`](phase10_implementation_plan.md) | **Release preparation** — the plan behind this group release (docs, de-personalization, licensing, cleanup). |
| [`phase8f_completion_plan.md`](phase8f_completion_plan.md) | Remaining AMIP-diffusion (Phase 8) punch list. |
| [`phase8e_midway3_checkpoint_inventory.md`](phase8e_midway3_checkpoint_inventory.md) | Diffusion checkpoint inventory (contains collaborator paths — review before wider sharing). |

> Phase status is summarized at the top / end of `implementation_plan.md`.
> Phase 9 (multi-cluster) is done; Phase 10 (this release) is in progress.

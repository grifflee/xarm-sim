# 10k lift visual checkpoint

Date: 2026-07-23

Artifact root: `/data/store/griffen_sim_mcaps/lift_10k_checkpoint`

The 10,000-episode run has **not** started. These artifacts are the required human gate.

## What to review

- `visual_batch/`: ten complete Madrona ray-traced + gsplat videos covering nearest and
  farthest spawn, both table-side edges, beside-base spawn, far/post-drop/broad arm starts,
  and two representative distribution draws. All ten completed successfully.
- `grasp_ab/seed100000_weld_vs_physical_side_by_side.mp4`: proximity-gated weld beside
  current-upstream-style physical grasp (`noslip_iterations=10`, no weld).
- `render_parity/nyx_raster_batch_contact_sheet.png`: labeled reset-frame comparison of
  Nyx, Genesis raster composite, and Madrona batch across the same ten scenes.
- `coverage/annulus_feasibility.png`: the measured full-table reach sweep with the selected
  `r=0.250–0.445 m` production annulus overlaid.
- `reports/`: MCAP distribution and joint-time reports from the three-episode full-resolution
  batch smoke test.
- `smoke_batch_100000/`: three 640x480 MCAPs. The format comparison and real-MCAP layout
  validation pass; all three demonstrations succeed.

Before 10k generation, grifflee must choose the renderer and either the proximity-gated
weld or physical/no-weld grasp after reviewing these artifacts.

# TODO — variation to add for sim2real transfer

Scoped against the ACTUAL deployment, which is a continuous loop, not episodes:

> The cube is placed near the middle of the table. The model is simply left running. Its job
> is to pick the cube up, carry it to the centre, and drop it. Wherever it lands — or
> wherever a human hand slides it to — the robot re-locates it and repeats, **without ever
> returning home**.

Nothing here blocks the current 10.5k batch. That batch has no lighting or physics variation
and is still the right thing to train first: it establishes whether the pipeline transfers at
all, and it is the baseline every randomization experiment below gets compared against.

## What the current batch actually varies

| axis | status |
|---|---|
| cube position | annulus r 0.25–0.55, x ≥ 0.115 (front-only), y −0.288…+0.20 |
| cube yaw | uniform ±45° (full space — a cube is 4-fold symmetric) |
| arm start | 4-bucket mixture, below |
| camera pose | ±15° / ±5 cm jitter on low + side, per episode |
| episode tempo | 0.85–1.30× segment timing |
| **lighting** | **none — fixed** (per-episode jitter now EXISTS but is off by default; see item 1) |
| **cube/robot appearance** | **none — fixed** |
| **physics (mass, friction)** | **none — fixed** |
| **actuation noise** | **none — perfect execution** |
| wrist camera mount | fixed (jitter exists but is unreviewed; every approved batch keeps it 0) |

Arm-start mixture, measured TCP spread over 80 episodes:

| bucket | weight | spread | note |
|---|---|---|---|
| `post_drop` | 0.40 | 3.3 cm mean | tight *by design* — the state right after a release. Correct: this is the most common real state. |
| `far` | 0.25 | 16.5 cm mean, 26 cm max | broadly randomized |
| `broad` | 0.25 | 20.0 cm mean, 33 cm max | broadly randomized |
| `home` | 0.10 | 1.4 cm mean | ±3° joint jitter only — effectively one pose |

So half the dataset starts broadly randomized and 40% starts clustered at the post-drop pose
(right), but **10% starts at a pose the deployment loop never visits** — `home` only occurs
at power-on. Candidate: drop to ~0.02 or redistribute into `post_drop`/`broad`.

## Two structural blockers found 2026-07-26

**Lighting cannot vary on the `batch` path at all.** ~~`src/xsim/batch_renderer.py`'s
`BatchConfig` hardcodes the Madrona rig — key 1.7, fill 0.85, fixed directions, no RNG. The
`nyx_light_*_jitter` config fields feed `_sample_appearance` for the **Nyx** renderer only.
Production uses `batch`. So the most obvious visual domain gap is currently un-randomizable
without code.~~ **RESOLVED 2026-07-26 — see item 1 below.** The rig turned out to be
mutable after `scene.build()`, so this is per-episode and free; no rebuild, no subprocess.

**Appearance randomization as built is prohibitively expensive.** Setting any
`APPEARANCE_JITTER_FIELDS` routes generation through
`_run_appearance_subprocess_batch` — "one Nyx subprocess per episode for fresh
lights/materials". At ~40 s of Genesis startup per episode that is days for 10k. This is why
the approved stack appearance recipe (2026-07-07) was only ever used on small batches.

## Prioritised work

### 1. Jitter the Madrona light rig — DONE 2026-07-26 (off by default)
Per-episode jitter on `BatchConfig`'s two lights: direction ±10–15°, intensity ±25%. Passed
into the existing rig at reset, so **no subprocess and no scene rebuild** — free at
generation time. The lab's lighting changes with time of day and the model has so far seen
exactly one lighting condition. Close this first.

Implemented as `--env.batch-light-dir-jitter-deg 12 --env.batch-light-intensity-jitter 0.25
--env.batch-shadow-strength-jitter 0.15` (all default 0 = the baseline rig, bit-for-bit).
Recorded per episode in the manifest under `lights` and `shadow_strength`.

**The axis that matters is table shadows, not robot brightness** (grifflee, 2026-07-26).
Direction jitter moves where the arm's shadow falls; `batch_shadow_strength_jitter` changes
how dark it is. That second one was the real gap: shadow darkness is *not* a function of
light intensity at all — `_composite_splat` renders the shadow SHAPE with Madrona and then
attenuates the baked splat tabletop by the fixed `batch_shadow_strength`, so every shadow in
every episode ever generated is exactly 45% dark. Intensity jitter is the least valuable of
the three (the white arm already clips 27% of its lit pixels at the nominal key).

**Physical bounds are enforced in code, not by the defaults being small.** The lab is lit
from the ceiling, so `batch_light_min_elevation_deg` (default 30°) clamps every jittered
light to at least that far below horizontal — verified at an absurd ±60° jitter, 40,000
draws, min elevation exactly 30.000° and nothing at or below the horizon. Shadow strength is
clamped to `BATCH_SHADOW_STRENGTH_LIMITS = (0.0, 0.65)`; 0 is a real fully-diffused ceiling,
but a pure-black umbra requires a lone point source in an unlit black room.

**The premise that the rig is frozen at construction was wrong, and the reason is worth
knowing.** `scene.add_light` is `@gs.assert_unbuilt` and `_add_cameras()` runs once, which
makes the rig *look* baked. It is not. Genesis' `BatchRenderer` keeps the lights as a plain
Python list, and `MadronaBatchRendererAdapter.init()` memcpys them into the `LightEntity` ECS
columns and re-runs Madrona's **RenderInit** task graph. The light entities themselves are
created in the `Sim` world constructor (`gs_madrona/src/bridge/sim.cpp`), *not* in a task
graph, so `init()` is re-entrant: it only sorts, copies and re-packs. Measured 1.1 ms per
call, zero GPU allocation growth over 60 calls, and restoring the nominal rig reproduces the
original frames pixel-for-pixel.

The trap to avoid: writing the exported light columns alone renders nothing new.
`lightUpdate` is only added to the graph when `update_visual_properties` is true
(`gs_madrona/src/render/ecs_system.cpp`), which holds for RenderInit but **not** for the
per-frame Render graph. `init()` is the only Python-reachable way to run it.

Known asymmetry, measured: the white xArm already clips 27% of its lit pixels at the nominal
key of 1.7, so +25% intensity buys mostly more clipping (41%) while −25% recovers shading
detail. If this wants widening later, widen the dark side or lower the key — don't raise it.
This constrains robot brightness only; shadows are dark regions and never clip.

**Blocker found while verifying the shadows — needs grifflee, no code changed.**
`batch_shadow_blur_px = 3.0` is an absolute pixel count tuned and approved at 640x480. The
2026-07-26 drop to 96x72 made it 6.7x wider relative to the frame, and it now smears the
shadow it is meant to soften. Same pose (seed 9002 step 150, side cam) at strength 0.45,
deepest tabletop darkening:

| render | blur | deepest | table below 0.85x |
|---|---|---|---|
| 640x480 | 3.00 px | 0.78x | 3.9% | (the approved look) |
| **96x72** | **3.00 px** | **0.84x** | **0.1%** | (production today) |
| 96x72 | 0.45 px | 0.67x | 4.6% | (res-scaled, restores it) |

So current 96x72 batches carry a markedly flatter shadow than the one signed off on
2026-07-23. The fix is to scale it with render width exactly as `STATIC_CAM_MARGIN_PX`
already does (that constant keeps an explicit `STATIC_CAM_MARGIN_REF_W = 640.0` for this
reason), but it changes an approved visual, so it is flagged rather than applied.
Panel: `outputs/sim_preview/batch_shadow_blur_resolution.png`.

### 2. Physics randomization — does not exist at all
No mass or friction jitter anywhere; the cube is always `friction=2.0` and one mass. For a
contact-rich frictional grasp this is a classic transfer failure — the policy learns one
closing behaviour tuned to one contact model. Suggest ±30% friction, ±20% mass per episode,
set at reset (no rebuild). Note `AGENTS.md` treats `gripper_grasp_dof: 0.58` and
`noslip_iterations: 10` as load-bearing; randomize the *object*, not those.

### 3. Actuation noise — does not exist
Sim tracks commanded joint positions perfectly; the real arm has tracking error and latency.
Small per-step joint-command noise stops the policy relying on exact execution.

### 4. Deployment-specific gaps (from the loop description)

- **The policy has never seen a human hand.** In deployment a hand reaches in and slides the
  cube — a large moving object entering frame, with zero training examples. Its behaviour
  when one appears is entirely unconstrained. Biggest effort of anything here; the
  MANO/human-data work may make a plausible rendered hand feasible.
- **The cube is never in motion.** It always spawns settled. When a human slides it the
  policy sees a *moving* cube, never trained on. Cheap approximation: give a fraction of
  episodes a small initial cube velocity so it is still sliding to rest in the first frames.
- **`home` starts (10%)** model a state that only occurs at power-on — see above.

### 5. Lower priority
- Wrist-mount jitter (`wrist_jitter_deg/cm`) is implemented but has never been reviewed; the
  mount is a verified guess, so jittering it needs a visual gate first.
- Distractor objects on the table. The splat background already carries real desk clutter, so
  probably low value.

## Suggested order

1–3 together are roughly half a day and attack the three axes that genuinely differ between
sim and the real cell: light, contact physics, actuation. Do them as one batch and compare
against the un-randomized 10.5k baseline. Item 4's hand is a separate project; the
moving-cube approximation is worth doing alongside 1–3.

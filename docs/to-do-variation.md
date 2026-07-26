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
| **lighting** | **none — fixed** |
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

**Lighting cannot vary on the `batch` path at all.** `src/xsim/batch_renderer.py`'s
`BatchConfig` hardcodes the Madrona rig — key 1.7, fill 0.85, fixed directions, no RNG. The
`nyx_light_*_jitter` config fields feed `_sample_appearance` for the **Nyx** renderer only.
Production uses `batch`. So the most obvious visual domain gap is currently un-randomizable
without code.

**Appearance randomization as built is prohibitively expensive.** Setting any
`APPEARANCE_JITTER_FIELDS` routes generation through
`_run_appearance_subprocess_batch` — "one Nyx subprocess per episode for fresh
lights/materials". At ~40 s of Genesis startup per episode that is days for 10k. This is why
the approved stack appearance recipe (2026-07-07) was only ever used on small batches.

## Prioritised work

### 1. Jitter the Madrona light rig — highest value, cheapest
Per-episode jitter on `BatchConfig`'s two lights: direction ±10–15°, intensity ±25%. Passed
into the existing rig at reset, so **no subprocess and no scene rebuild** — free at
generation time. The lab's lighting changes with time of day and the model has so far seen
exactly one lighting condition. Close this first.

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

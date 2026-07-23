# Batch-renderer splat shadow catcher

Date: 2026-07-23

## Problem

The batch path renders live robot/cube geometry with Madrona, then composites it over the
calibrated gsplat room. The simulated table is intentionally transparent because the
splat supplies the real table pixels. Although Madrona's key light has
`castshadow=true`, a baked splat is not geometry and cannot receive a dynamic shadow.
The first full checkpoint therefore had foreground lighting but no tabletop shadows.

Making the simulated table visible proved that Madrona was producing shadows correctly,
but replaced the calibrated table with a flat gray slab. The defect was in compositing,
not the light or shadow-ray implementation.

## Method

For batch + gsplat + transparent-slab rendering, `TaskEnv` adds a 2 mm thick,
visual-only neutral receiver immediately below the physical table plane. It has no
collision role and never appears directly in the final image.

Madrona renders the scene and its segmentation map together. The compositor uses the
receiver's unique segmentation ID to:

1. measure receiver luminance per pixel;
2. normalize it against the receiver's unshadowed 95th-percentile luminance;
3. convert the deficit into a shadow-attenuation mask;
4. soften the mask with a segmentation-normalized Gaussian blur; and
5. multiply only the original splat pixels inside the receiver footprint by that mask.

Robot, cube, and decor pixels remain normal Madrona foreground; background and table
texture remain the original aligned splat. Defaults are `batch_shadow_strength=0.45` and
`batch_shadow_blur_px=3.0`.

## Approval

The controlled fixed-camera comparison is
`shadow_audit/seed100000_before_after_shadows.mp4` under the 10k checkpoint artifact
root. The top panel is the original baked table and the bottom panel is the shadow
catcher. Grifflee approved the bottom panel on 2026-07-23.

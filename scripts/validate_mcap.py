"""Inspect and compare Foxglove protobuf MCAP episode files.

MCAP files are logs, so their "shape" is topic -> schema -> message count -> payload
shape. This script prints that summary and decodes the first message for the training
relevant real robot topics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

import cv2
import numpy as np
from numcodecs import Zstd

CORE_TOPICS = {
    "/cam/low/image_raw",
    "/cam/side/image_raw",
    "/camera/camera/color/image_raw/compressed",
    "/xarm/joint_states",
    "/xarm/robot_states",
    "/xgym/gripper",
}

CALIBRATION_TOPIC_SCHEMAS = {
    "/cam/low/camera_info": "foxglove.CameraCalibration",
    "/cam/side/camera_info": "foxglove.CameraCalibration",
    "/camera/camera/color/camera_info": "foxglove.CameraCalibration",
    "/tf": "foxglove.FrameTransform",
}

CALIBRATION_TOPIC_COUNTS = {
    "/cam/low/camera_info": 1,
    "/cam/side/camera_info": 1,
    "/camera/camera/color/camera_info": 1,
    "/tf": 3,
}

# Lift approach timing scales with start-to-first-waypoint distance, floor 1x and
# cap 2x; other segments remain fixed. MCAP records through a 0.3 s release tail,
# decimated from 120 Hz to 30 Hz. Keep the approved truncation/runaway margins around
# the re-derived design envelope. Stack retains its fixed timing.
TASK_SEGMENT_WEIGHTS = {
    "lift": (2.6, 0.8, 0.8, 0.4, 1.0, 0.6),
    "stack": (2.6, 0.8, 0.8, 0.4, 1.0, 0.8, 0.6),  # xsim.scripted_stack_policy
}
DEFAULT_STEPS_PER_SEGMENT = 108
TEMPO_RANGE = (0.85, 1.30)
DEFAULT_RELEASE_TAIL_S = 0.3
DEFAULT_PHYSICS_DT = 1.0 / 120.0
DEFAULT_RECORD_EVERY = 4
LIFT_APPROACH_SCALE_CAP = 2.0
# margins below/above the design envelope; chosen so the lift gate stays exactly the
# verified 115..240 from the approved batches (design 152..227)
FRAME_MARGIN_LOW = 37
FRAME_MARGIN_HIGH = 13


def _recorded_frames(
    steps_per_segment: int,
    weights: tuple[float, ...],
    approach_scale: float = 1.0,
) -> int:
    segment_steps = [
        max(2, round(w * steps_per_segment * (approach_scale if i == 0 else 1.0)))
        for i, w in enumerate(weights)
    ]
    release_step = 1 + sum(segment_steps)
    release_tail = max(1, round(DEFAULT_RELEASE_TAIL_S / DEFAULT_PHYSICS_DT))
    record_until = release_step + release_tail
    return (record_until - 1) // DEFAULT_RECORD_EVERY + 1


def design_frame_envelope(task: str = "lift") -> tuple[int, int]:
    weights = TASK_SEGMENT_WEIGHTS[task]
    max_approach_scale = LIFT_APPROACH_SCALE_CAP if task == "lift" else 1.0
    return (
        _recorded_frames(round(DEFAULT_STEPS_PER_SEGMENT * TEMPO_RANGE[0]), weights),
        _recorded_frames(
            round(DEFAULT_STEPS_PER_SEGMENT * TEMPO_RANGE[1]),
            weights,
            max_approach_scale,
        ),
    )


def core_frame_gate(task: str = "lift") -> tuple[int, int]:
    lo, hi = design_frame_envelope(task)
    return lo - FRAME_MARGIN_LOW, hi + FRAME_MARGIN_HIGH


def _str(b: bytes, o: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", b, o)
    o += 4
    return b[o : o + n].decode("utf-8", "replace"), o + n


def read_records(path: str):
    """Yield (op, body) for every top-level record, transparently unchunking zstd/none."""
    with open(path, "rb") as f:
        f.seek(8)  # skip magic
        data = f.read()
    o = 0
    while o < len(data) - 8:
        op = data[o]
        (ln,) = struct.unpack_from("<Q", data, o + 1)
        body = data[o + 9 : o + 9 + ln]
        o += 9 + ln
        if op == 0x02:  # footer
            break
        if op == 0x06:  # chunk
            p = 28
            comp, p = _str(body, p)
            (cs,) = struct.unpack_from("<Q", body, p)
            p += 8
            rec = Zstd().decode(body[p : p + cs]) if comp == "zstd" else body[p : p + cs]
            q = 0
            while q < len(rec):
                rop = rec[q]
                (rln,) = struct.unpack_from("<Q", rec, q + 1)
                yield rop, rec[q + 9 : q + 9 + rln]
                q += 9 + rln
        else:
            yield op, body


def scan(path: str):
    chans: dict[int, tuple[str, int]] = {}
    schemas: dict[int, str] = {}
    counts: dict[int, int] = {}
    first_msg: dict[int, bytes] = {}
    for op, body in read_records(path):
        if op == 0x03:  # schema
            (sid,) = struct.unpack_from("<H", body, 0)
            name, _ = _str(body, 2)
            schemas[sid] = name
        elif op == 0x04:  # channel
            (cid, sid) = struct.unpack_from("<HH", body, 0)
            topic, _ = _str(body, 4)
            chans[cid] = (topic, sid)
        elif op == 0x05:  # message
            (cid,) = struct.unpack_from("<H", body, 0)
            counts[cid] = counts.get(cid, 0) + 1
            first_msg.setdefault(cid, body[22:])
    return chans, schemas, counts, first_msg


def schema_by_topic(chans: dict[int, tuple[str, int]], schemas: dict[int, str]) -> dict[str, str]:
    return {topic: schemas.get(sid, "?") for _, (topic, sid) in chans.items()}


def counts_by_topic(chans: dict[int, tuple[str, int]], counts: dict[int, int]) -> dict[str, int]:
    return {topic: counts.get(cid, 0) for cid, (topic, _) in chans.items()}


def compare_topic_layout(
    chans: dict[int, tuple[str, int]],
    schemas: dict[int, str],
    counts: dict[int, int],
    ref_chans: dict[int, tuple[str, int]],
    ref_schemas: dict[int, str],
    task: str = "lift",
) -> list[str]:
    """Format gate for sim MCAPs vs real lift MCAPs.

    The six training streams must match the real reference topics/schemas. Sim files
    are also expected to carry four ground-truth calibration topics written once at
    episode start;
    those are allowlisted extras, not a free pass for arbitrary new topics.
    """
    ref_topics = schema_by_topic(ref_chans, ref_schemas)
    have_topics = schema_by_topic(chans, schemas)
    have_counts = counts_by_topic(chans, counts)
    problems: list[str] = []

    missing_ref_core = sorted(CORE_TOPICS - set(ref_topics))
    if missing_ref_core:
        problems.append(f"reference missing core topics: {missing_ref_core}")

    missing_core = sorted(CORE_TOPICS - set(have_topics))
    if missing_core:
        problems.append(f"missing core topics: {missing_core}")

    extra_unallowed = sorted(set(have_topics) - CORE_TOPICS - set(CALIBRATION_TOPIC_SCHEMAS))
    if extra_unallowed:
        problems.append(f"unallowlisted extra topics: {extra_unallowed}")

    for topic in sorted(CORE_TOPICS & set(have_topics) & set(ref_topics)):
        if have_topics[topic] != ref_topics[topic]:
            problems.append(f"schema mismatch on {topic}: {have_topics[topic]} != {ref_topics[topic]}")

    missing_cal = sorted(set(CALIBRATION_TOPIC_SCHEMAS) - set(have_topics))
    if missing_cal:
        problems.append(f"missing calibration topics: {missing_cal}")

    for topic, expected_schema in sorted(CALIBRATION_TOPIC_SCHEMAS.items()):
        if topic in have_topics and have_topics[topic] != expected_schema:
            problems.append(f"schema mismatch on {topic}: {have_topics[topic]} != {expected_schema}")

    for topic, expected_count in sorted(CALIBRATION_TOPIC_COUNTS.items()):
        if topic in have_counts and have_counts[topic] != expected_count:
            problems.append(f"{topic}: expected {expected_count} message(s), got {have_counts[topic]}")

    if not missing_core:
        core_counts = {topic: have_counts.get(topic, 0) for topic in sorted(CORE_TOPICS)}
        lo, hi = min(core_counts.values()), max(core_counts.values())
        gate_lo, gate_hi = core_frame_gate(task)
        design_lo, design_hi = design_frame_envelope(task)
        if lo < gate_lo or hi > gate_hi:
            problems.append(
                f"core message count out of range: {lo}..{hi} "
                f"(expected {gate_lo}..{gate_hi} for task {task!r}; "
                f"design envelope {design_lo}..{design_hi})"
            )

    return problems


def _varint(b: bytes, o: int) -> tuple[int, int]:
    v = s = 0
    while True:
        x = b[o]
        o += 1
        v |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return v, o


def _read_fields(buf: bytes) -> dict[int, list[object]]:
    o = 0
    out: dict[int, list[object]] = {}
    while o < len(buf):
        tag, o = _varint(buf, o)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            value, o = _varint(buf, o)
        elif wt == 1:
            (value,) = struct.unpack_from("<d", buf, o)
            o += 8
        elif wt == 2:
            n, o = _varint(buf, o)
            value = buf[o : o + n]
            o += n
        elif wt == 5:
            (value,) = struct.unpack_from("<I", buf, o)
            o += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        out.setdefault(fn, []).append(value)
    return out


def _one(fields: dict[int, list[object]], number: int, default=None):
    values = fields.get(number)
    return values[0] if values else default


def _decode_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def parse_rawimage(buf: bytes) -> dict:
    """Foxglove RawImage fields: timestamp=1 width=2 height=3 encoding=4 step=5 data=6 frame_id=7."""
    fields = _read_fields(buf)
    return {
        "width": int(_one(fields, 2, 0)),
        "height": int(_one(fields, 3, 0)),
        "encoding": _decode_str(_one(fields, 4, b"")),
        "step": int(_one(fields, 5, 0)),
        "data": _one(fields, 6, b""),
        "frame_id": _decode_str(_one(fields, 7, b"")),
    }


def _vec3(buf: bytes) -> tuple[float, float, float]:
    fields = _read_fields(buf)
    return (float(_one(fields, 1, 0.0)), float(_one(fields, 2, 0.0)), float(_one(fields, 3, 0.0)))


def _quat(buf: bytes) -> tuple[float, float, float, float]:
    fields = _read_fields(buf)
    return (
        float(_one(fields, 1, 0.0)),
        float(_one(fields, 2, 0.0)),
        float(_one(fields, 3, 0.0)),
        float(_one(fields, 4, 0.0)),
    )


def parse_pose(buf: bytes) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    fields = _read_fields(buf)
    pos = _vec3(_one(fields, 1, b""))
    quat = _quat(_one(fields, 2, b""))
    return pos, quat


def parse_joint_states(buf: bytes) -> list[dict[str, float | str]]:
    fields = _read_fields(buf)
    joints = []
    for raw in fields.get(2, []):
        jf = _read_fields(raw)
        joints.append(
            {
                "name": _decode_str(_one(jf, 1, b"")),
                "position": float(_one(jf, 2, 0.0)),
                "velocity": float(_one(jf, 3, 0.0)),
                "effort": float(_one(jf, 5, 0.0)),
            }
        )
    return joints


def parse_gripper(buf: bytes) -> dict[str, float]:
    fields = _read_fields(buf)
    return {
        "rad": float(_one(fields, 2, 0.0)),
        "norm": float(_one(fields, 3, 0.0)),
        "raw": float(_one(fields, 4, 0.0)),
    }


def rawimage_summary(topic: str, msg: bytes) -> str:
    ri = parse_rawimage(msg)
    w, h, step, enc = ri["width"], ri["height"], ri["step"], ri["encoding"]
    data = ri["data"]
    shape = "?"
    mean = "?"
    ok = "??"
    try:
        if enc == "yuv422_yuy2":
            arr = np.frombuffer(data, np.uint8).reshape(h, w, 2)
            rgb = cv2.cvtColor(arr, cv2.COLOR_YUV2RGB_YUY2)
            shape = f"uint8[{h},{w},2]"
            mean = f"{rgb.mean():.0f}"
            ok = "OK" if step == w * 2 and len(data) == w * h * 2 else "??"
        elif enc == "rgb8":
            rgb = np.frombuffer(data, np.uint8).reshape(h, w, 3)
            shape = f"uint8[{h},{w},3]"
            mean = f"{rgb.mean():.0f}"
            ok = "OK" if step == w * 3 and len(data) == w * h * 3 else "??"
    except Exception as exc:  # noqa: BLE001 - this is an inspection script
        mean = f"decode-error:{exc}"
    return (
        f"  [{ok}] {topic:45s} {w}x{h} step={step} enc={enc} "
        f"shape={shape} frame_id={ri['frame_id']!r} rgb_mean={mean}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mcap", type=Path)
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--task", choices=sorted(TASK_SEGMENT_WEIGHTS), default="lift",
                    help="protocol whose frame-count gate applies (default: lift)")
    args = ap.parse_args()

    chans, schemas, counts, first = scan(str(args.mcap))
    by_topic = {t: (cid, sid) for cid, (t, sid) in chans.items()}

    print(f"== {args.mcap} ==")
    print(f"{'topic':45s} {'schema':24s} count")
    for cid, (topic, sid) in sorted(chans.items(), key=lambda kv: kv[1][0]):
        print(f"{topic:45s} {schemas.get(sid, '?'):24s} {counts.get(cid, 0)}")

    print()
    print("-- RawImage payloads --")
    for cid, (topic, sid) in sorted(chans.items(), key=lambda kv: kv[1][0]):
        if schemas.get(sid) == "foxglove.RawImage":
            print(rawimage_summary(topic, first[cid]))

    print()
    print("-- Proprio samples --")
    for topic in ("/xarm/joint_states", "/xarm/robot_states", "/xgym/gripper"):
        if topic not in by_topic:
            continue
        cid, sid = by_topic[topic]
        schema = schemas.get(sid)
        if schema == "foxglove.JointStates":
            joints = parse_joint_states(first[cid])
            names = [j["name"] for j in joints]
            positions = [round(float(j["position"]), 4) for j in joints]
            print(f"  {topic}: names={names} position={positions}")
        elif schema == "foxglove.Pose":
            pos, quat = parse_pose(first[cid])
            print(f"  {topic}: position={tuple(round(v, 3) for v in pos)} quat={tuple(round(v, 4) for v in quat)}")
        elif schema == "xclients.Gripper":
            print(f"  {topic}: {parse_gripper(first[cid])}")

    print()
    print("-- Calibration topics --")
    topic_counts = counts_by_topic(chans, counts)
    for topic, expected_schema in sorted(CALIBRATION_TOPIC_SCHEMAS.items()):
        if topic not in by_topic:
            print(f"  [missing] {topic}")
            continue
        _, sid = by_topic[topic]
        print(f"  {topic}: schema={schemas.get(sid, '?')} count={topic_counts.get(topic, 0)}")

    if args.reference and args.reference.exists():
        ref_chans, ref_schemas, _, _ = scan(str(args.reference))
        problems = compare_topic_layout(chans, schemas, counts, ref_chans, ref_schemas, task=args.task)
        print()
        print(f"-- vs reference {args.reference.name} --")
        if problems:
            print("  layout: FAIL")
            for problem in problems:
                print(f"  - {problem}")
            raise SystemExit(1)
        print("  layout: PASS (core topics match reference; calibration topics present)")


if __name__ == "__main__":
    main()

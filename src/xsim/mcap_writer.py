"""Write simulated episodes as Foxglove protobuf MCAP matching real robot data.

The real lift episodes under /data/store/mcaps/single/lift are MCAP conversions of the
bela zarr recordings. Their training-relevant layout is:

- /cam/low/image_raw                         foxglove.RawImage, yuv422_yuy2
- /cam/side/image_raw                        foxglove.RawImage, yuv422_yuy2
- /camera/camera/color/image_raw/compressed  foxglove.RawImage, rgb8
- /xarm/joint_states                         foxglove.JointStates
- /xarm/robot_states                         foxglove.Pose
- /xgym/gripper                              xclients.Gripper

Despite the RealSense topic name ending in "compressed", the MCAP payload is decoded raw
RGB, matching xclients/scripts/zarr_to_foxglove_mcap.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import foxglove
from foxglove.channels import (
    CameraCalibrationChannel,
    FrameTransformChannel,
    JointStatesChannel,
    PoseChannel,
    RawImageChannel,
)
from foxglove.messages import (
    CameraCalibration,
    FrameTransform,
    JointState,
    JointStates,
    Pose,
    Quaternion,
    RawImage,
    Timestamp,
    Vector3,
)
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2

YUYV_ENCODING = "yuv422_yuy2"
RGB_ENCODING = "rgb8"
WRIST_TOPIC = "/camera/camera/color/image_raw/compressed"
JOINT_NAMES: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")


def topic_for_camera(name: str) -> str:
    return WRIST_TOPIC if name == "wrist" else f"/cam/{name}/image_raw"


def calibration_topic_for_camera(name: str) -> str:
    return "/camera/camera/color/camera_info" if name == "wrist" else f"/cam/{name}/camera_info"


def optical_frame_for_camera(name: str) -> str:
    return f"cam_{name}_optical"


def encoding_for_camera(name: str) -> str:
    return RGB_ENCODING if name == "wrist" else YUYV_ENCODING


def rgb_to_yuyv(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8[H,W,3] -> packed YUYV422 uint8[H,W,2] (yuy2)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected [H,W,3] rgb, got {rgb.shape}")
    h, w = rgb.shape[:2]
    if w % 2:
        raise ValueError(f"YUYV needs even width, got {w}")
    yuv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YUV)
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    out = np.empty((h, w, 2), dtype=np.uint8)
    out[..., 0] = y
    u_pair = ((u[:, 0::2].astype(np.uint16) + u[:, 1::2]) // 2).astype(np.uint8)
    v_pair = ((v[:, 0::2].astype(np.uint16) + v[:, 1::2]) // 2).astype(np.uint8)
    out[:, 0::2, 1] = u_pair
    out[:, 1::2, 1] = v_pair
    return out


def _timestamp(stamp_ns: int) -> Timestamp:
    sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
    return Timestamp(sec=sec, nsec=nsec)


def _quat_wxyz_from_rot(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix → quaternion (w,x,y,z), Shepperd's method."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return float(w), float(x), float(y), float(z)


# /xgym/gripper: replicate xclients.messages.Gripper without importing xclients.
_GRIPPER_TYPE = "xclients.Gripper"


def _gripper_file_descriptor() -> "descriptor_pb2.FileDescriptorProto":
    proto = descriptor_pb2.FileDescriptorProto()
    proto.name = "xclients/gripper.proto"
    proto.package = "xclients"
    proto.syntax = "proto2"
    proto.dependency.append("google/protobuf/timestamp.proto")
    msg = proto.message_type.add()
    msg.name = "Gripper"

    f = msg.field.add()
    f.name, f.number = "timestamp", 1
    f.label = descriptor_pb2.FieldDescriptorProto.LABEL_REQUIRED
    f.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    f.type_name = ".google.protobuf.Timestamp"

    for number, name in ((2, "rad"), (3, "norm"), (4, "raw")):
        f = msg.field.add()
        f.name, f.number = name, number
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
    return proto


def _gripper_descriptor_set() -> bytes:
    ds = descriptor_pb2.FileDescriptorSet()
    ds.file.add().ParseFromString(timestamp_pb2.DESCRIPTOR.serialized_pb)
    ds.file.add().CopyFrom(_gripper_file_descriptor())
    return ds.SerializeToString()


def _gripper_message_cls():
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
    pool.Add(_gripper_file_descriptor())
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(_GRIPPER_TYPE))


GRIPPER_SCHEMA = foxglove.Schema(name=_GRIPPER_TYPE, encoding="protobuf", data=_gripper_descriptor_set())
_GRIPPER_CLS = _gripper_message_cls()


def encode_gripper(stamp_ns: int, norm: float) -> bytes:
    """Serialize a gripper sample. norm in [0,1], where 1=open and 0=closed."""
    msg = _GRIPPER_CLS(rad=norm * 0.85, norm=norm, raw=norm * 850.0)
    msg.timestamp.seconds = int(stamp_ns) // 1_000_000_000
    msg.timestamp.nanos = int(stamp_ns) % 1_000_000_000
    return msg.SerializeToString()


@dataclass
class CameraSpec:
    """Per-camera MCAP output settings."""

    name: str
    width: int = 640
    height: int = 480
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    topic: str | None = None
    encoding: str | None = None
    frame_id: str = ""

    @property
    def topic_name(self) -> str:
        return self.topic or topic_for_camera(self.name)

    @property
    def raw_encoding(self) -> str:
        return self.encoding or encoding_for_camera(self.name)


class EpisodeMcapWriter:
    """Streams one episode to an MCAP. Use as a context manager and call log_step."""

    def __init__(
        self,
        path: str | Path,
        cameras: dict[str, CameraSpec],
        *,
        joint_names: tuple[str, ...] = JOINT_NAMES,
        allow_overwrite: bool = True,
    ) -> None:
        self.path = str(path)
        self.cameras = cameras
        self.joint_names = tuple(joint_names)
        self._allow_overwrite = allow_overwrite
        self._mcap = None
        self._img: dict[str, RawImageChannel] = {}
        self._joints: JointStatesChannel | None = None
        self._robot_states: PoseChannel | None = None
        self._gripper = None
        # Optional /teacher/* channels mirroring the measured ones. Created lazily on the
        # first step that supplies teacher data, so an episode without teacher labels
        # never instantiates them and writes byte-identical MCAPs to today.
        self._teacher_joints: JointStatesChannel | None = None
        self._teacher_robot_states: PoseChannel | None = None
        self._teacher_gripper = None

    def __enter__(self) -> "EpisodeMcapWriter":
        self._mcap = foxglove.open_mcap(self.path, allow_overwrite=self._allow_overwrite)
        for name, spec in self.cameras.items():
            self._img[name] = RawImageChannel(spec.topic_name)
        self._joints = JointStatesChannel("/xarm/joint_states")
        self._robot_states = PoseChannel("/xarm/robot_states")
        self._gripper = foxglove.Channel("/xgym/gripper", schema=GRIPPER_SCHEMA, message_encoding="protobuf")
        self._calib = {name: CameraCalibrationChannel(calibration_topic_for_camera(name)) for name in self.cameras}
        self._tf = FrameTransformChannel("/tf")
        return self

    def log_calibration(self, stamp_ns: int, extrinsics: dict[str, np.ndarray]) -> None:
        """Write ground-truth camera intrinsics + extrinsics once per episode.

        Intrinsics: foxglove.CameraCalibration per camera from the CameraSpec (zero
        distortion — the sim cameras are ideal pinholes). Extrinsics: FrameTransforms on
        /tf, ``base → cam_<name>_optical`` for static cameras (the episode's actual
        jittered camera-to-base pose, OpenCV optical convention) and
        ``link_tcp → cam_wrist_optical`` for the wrist mount. Keys in ``extrinsics``
        follow TaskEnv.episode_extrinsics: "low", "side", "wrist_mount".
        """
        ts = _timestamp(stamp_ns)
        for name, spec in self.cameras.items():
            k = [spec.fx, 0.0, spec.cx, 0.0, spec.fy, spec.cy, 0.0, 0.0, 1.0]
            self._calib[name].log(
                CameraCalibration(
                    timestamp=ts,
                    frame_id=optical_frame_for_camera(name),
                    width=spec.width,
                    height=spec.height,
                    distortion_model="plumb_bob",
                    D=[0.0] * 5,
                    K=k,
                    R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    P=[k[0], k[1], k[2], 0.0, k[3], k[4], k[5], 0.0, k[6], k[7], k[8], 0.0],
                ),
                log_time=int(stamp_ns),
            )
        for key, T in extrinsics.items():
            T = np.asarray(T, dtype=np.float64)
            if key.endswith("_mount"):
                parent, child = "link_tcp", optical_frame_for_camera(key[: -len("_mount")])
            else:
                parent, child = "base", optical_frame_for_camera(key)
            qw, qx, qy, qz = _quat_wxyz_from_rot(T[:3, :3])
            self._tf.log(
                FrameTransform(
                    timestamp=ts,
                    parent_frame_id=parent,
                    child_frame_id=child,
                    translation=Vector3(x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3])),
                    rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
                ),
                log_time=int(stamp_ns),
            )

    def __exit__(self, *exc) -> None:
        if self._mcap is not None:
            self._mcap.__exit__(*exc)
            self._mcap = None

    def log_step(
        self,
        stamp_ns: int,
        images: dict[str, np.ndarray],
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_eff: np.ndarray,
        extrinsics: dict[str, np.ndarray] | None,
        ee_pose: np.ndarray,
        gripper_norm: float,
        teacher_joint_pos: np.ndarray | None = None,
        teacher_ee_pose: np.ndarray | None = None,
        teacher_gripper_norm: float | None = None,
    ) -> None:
        """Log one recorded timestep.

        images[name] is RGB uint8[H,W,3]. ee_pose is [x,y,z,qw,qx,qy,qz] in metres.
        Real xArm MCAP Pose positions are in millimetres, so TCP xyz is scaled by 1000.

        When teacher data is supplied (all three teacher_* args), three parallel channels
        record the expert's label for DAgger — ``/teacher/joint_states`` (7 commanded joint
        targets, zero velocity/effort), ``/teacher/robot_states`` (commanded EE pose, same
        x1000 m->mm scaling as ``/xarm/robot_states``; teacher_ee_pose is [x,y,z,qw,qx,qy,qz]
        in metres) and ``/teacher/gripper`` (commanded norm, 1=open/0=closed). Omit them and
        the MCAP is byte-identical to a measured-only recording.
        """
        ts = _timestamp(stamp_ns)

        for name, spec in self.cameras.items():
            rgb = np.ascontiguousarray(images[name][..., :3]).astype(np.uint8)
            height, width = rgb.shape[:2]
            encoding = spec.raw_encoding
            if encoding == YUYV_ENCODING:
                payload = rgb_to_yuyv(rgb).tobytes()
                step = width * 2
            elif encoding == RGB_ENCODING:
                payload = rgb.tobytes()
                step = int(rgb.strides[0])
            else:
                raise ValueError(f"unsupported RawImage encoding for {name}: {encoding!r}")

            self._img[name].log(
                RawImage(
                    timestamp=ts,
                    frame_id=spec.frame_id,
                    width=width,
                    height=height,
                    encoding=encoding,
                    step=step,
                    data=payload,
                ),
                log_time=int(stamp_ns),
            )

        joints = [
            JointState(name=n, position=float(p), velocity=float(v), effort=float(e))
            for n, p, v, e in zip(self.joint_names, joint_pos, joint_vel, joint_eff, strict=True)
        ]
        self._joints.log(JointStates(timestamp=ts, joints=joints), log_time=int(stamp_ns))

        px, py, pz, qw, qx, qy, qz = (float(v) for v in ee_pose)
        self._robot_states.log(
            Pose(
                position=Vector3(x=px * 1000.0, y=py * 1000.0, z=pz * 1000.0),
                orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
            ),
            log_time=int(stamp_ns),
        )

        self._gripper.log(encode_gripper(int(stamp_ns), float(gripper_norm)), log_time=int(stamp_ns))

        has_teacher = (
            teacher_joint_pos is not None
            and teacher_ee_pose is not None
            and teacher_gripper_norm is not None
        )
        if not has_teacher:
            return
        if self._teacher_joints is None:  # lazy: first teacher step creates the channels
            self._teacher_joints = JointStatesChannel("/teacher/joint_states")
            self._teacher_robot_states = PoseChannel("/teacher/robot_states")
            self._teacher_gripper = foxglove.Channel(
                "/teacher/gripper", schema=GRIPPER_SCHEMA, message_encoding="protobuf")

        t_joints = [
            JointState(name=n, position=float(p), velocity=0.0, effort=0.0)
            for n, p in zip(self.joint_names, np.asarray(teacher_joint_pos).reshape(-1), strict=True)
        ]
        self._teacher_joints.log(JointStates(timestamp=ts, joints=t_joints), log_time=int(stamp_ns))

        tpx, tpy, tpz, tqw, tqx, tqy, tqz = (float(v) for v in np.asarray(teacher_ee_pose).reshape(-1))
        self._teacher_robot_states.log(
            Pose(
                position=Vector3(x=tpx * 1000.0, y=tpy * 1000.0, z=tpz * 1000.0),
                orientation=Quaternion(x=tqx, y=tqy, z=tqz, w=tqw),
            ),
            log_time=int(stamp_ns),
        )

        self._teacher_gripper.log(
            encode_gripper(int(stamp_ns), float(teacher_gripper_norm)), log_time=int(stamp_ns))

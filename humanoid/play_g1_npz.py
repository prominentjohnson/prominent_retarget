#!/usr/bin/env python3
"""Kinematically replay a RoboNaldo G1 motion in MuJoCo's official viewer."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# The desktop on this machine is driven by the Intel GPU while MuJoCo's GLFW
# window works through NVIDIA PRIME offload. Respect explicit user settings,
# but make the editor's parameterless Run button work out of the box.
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit(
        "MuJoCo is not installed for this Python interpreter.\n"
        "Activate the project venv with: source .venv/bin/activate"
    ) from exc


# Exact NPZ column order documented and used by RoboNaldo_Deploy's FreeKick
# and BeyondMimic policies. Isaac Lab / PhysX stores articulation joints in
# this breadth-first, left/right-interleaved order. Do not replace it with
# MuJoCo's depth-first XML order: every value is assigned by name below.
ROBONALDO_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# RoboNaldo's converter writes Isaac Lab's root body first. Its source model's
# root is pelvis. The converter changes input xyzw to wxyz before logging the
# body quaternions; motions/README.md documents the saved order as wxyz too.
ROBONALDO_ROOT_BODY_NAME = "pelvis"
ROBONALDO_QUAT_ORDER = "wxyz"

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_NPZ = PROJECT_DIR / "third_party/RoboNaldo/motions/right_kick.npz"
DEFAULT_MODEL = (
    PROJECT_DIR
    / "third_party/RoboNaldo/RoboNaldo_Deploy/g1_description/scene.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinematically replay a RoboNaldo G1 NPZ with the official MuJoCo viewer."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=DEFAULT_NPZ,
        help=f"RoboNaldo motion NPZ (default: {DEFAULT_NPZ})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"G1 MJCF/XML scene (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop until the viewer is closed (default: enabled; use --no-loop to disable)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    parser.add_argument(
        "--fixed-root",
        action="store_true",
        help="Keep the model's initial root pose and replay only the 29 joints",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate and print the NPZ/model mapping without opening a viewer",
    )
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be a positive finite number")
    return args


def scalar_fps(value: np.ndarray) -> float:
    flat = np.asarray(value).reshape(-1)
    if flat.size != 1:
        raise ValueError(f"fps must contain exactly one value, got shape {value.shape}")
    fps = float(flat[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps!r}")
    return fps


def inspect_npz(path: Path) -> tuple[dict[str, np.ndarray], float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    print(f"NPZ: {path.resolve()}")
    print(f"keys: {list(arrays)}")
    for key, value in arrays.items():
        suffix = f", value={value.reshape(-1)[0]!r}" if value.size == 1 else ""
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}{suffix}")

    required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "fps")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"NPZ is missing required key(s): {missing}")

    fps = scalar_fps(arrays["fps"])
    joint_pos = arrays["joint_pos"]
    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be 2-D, got {joint_pos.shape}")
    frames = joint_pos.shape[0]
    if frames == 0:
        raise ValueError("joint_pos has no frames")
    if joint_pos.shape[1] != len(ROBONALDO_G1_JOINT_NAMES):
        raise ValueError(
            f"joint_pos has {joint_pos.shape[1]} columns; RoboNaldo G1 requires "
            f"{len(ROBONALDO_G1_JOINT_NAMES)}"
        )
    if not np.isfinite(joint_pos).all():
        raise ValueError("joint_pos contains NaN or infinity")

    if arrays["joint_vel"].shape != joint_pos.shape:
        raise ValueError(
            f"joint_vel shape {arrays['joint_vel'].shape} does not match joint_pos {joint_pos.shape}"
        )
    if arrays["body_pos_w"].shape != (frames, 30, 3):
        raise ValueError(f"expected body_pos_w shape ({frames}, 30, 3), got {arrays['body_pos_w'].shape}")
    if arrays["body_quat_w"].shape != (frames, 30, 4):
        raise ValueError(f"expected body_quat_w shape ({frames}, 30, 4), got {arrays['body_quat_w'].shape}")

    root_pos = arrays["body_pos_w"][:, 0, :]
    root_quat = arrays["body_quat_w"][:, 0, :]
    if not np.isfinite(root_pos).all() or not np.isfinite(root_quat).all():
        raise ValueError("root position/quaternion contains NaN or infinity")
    quat_norm = np.linalg.norm(root_quat, axis=1)
    if np.max(np.abs(quat_norm - 1.0)) > 1e-3:
        raise ValueError(
            "body_quat_w[:, 0] is not normalized "
            f"(norm range {quat_norm.min():.6f}..{quat_norm.max():.6f})"
        )

    print(f"fps: {fps:g}")
    print(f"frames: {frames}")
    print(f"sample span: {(frames - 1) / fps:.6f} s")
    print(f"loop period: {frames / fps:.6f} s")
    print(f"root source: body index 0 = {ROBONALDO_ROOT_BODY_NAME}")
    print(f"NPZ quaternion order: {ROBONALDO_QUAT_ORDER}")
    return arrays, fps, frames


def object_name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"<unnamed:{index}>"


def inspect_model(
    path: Path,
) -> tuple[mujoco.MjModel, list[int], tuple[int, int] | None]:
    if not path.is_file():
        raise FileNotFoundError(f"model not found: {path}")
    model = mujoco.MjModel.from_xml_path(str(path.resolve()))

    all_joint_names = [object_name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    free_ids = [
        i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    ]
    scalar_names = [name for i, name in enumerate(all_joint_names) if i not in free_ids]
    expected = list(ROBONALDO_G1_JOINT_NAMES)
    missing = [name for name in expected if name not in scalar_names]
    extra = [name for name in scalar_names if name not in expected]

    print(f"model: {path.resolve()}")
    print(f"nq={model.nq}, nv={model.nv}, njnt={model.njnt}, nbody={model.nbody}")
    print(f"NPZ joints ({len(expected)}): {expected}")
    print(f"MuJoCo scalar joints ({len(scalar_names)}): {scalar_names}")
    if missing or extra:
        print(f"missing in MuJoCo: {missing}", file=sys.stderr)
        print(f"extra in MuJoCo: {extra}", file=sys.stderr)
        raise ValueError("NPZ/MuJoCo joint-name mismatch; refusing to guess")

    qpos_addresses: list[int] = []
    print("explicit joint mapping (NPZ column -> MuJoCo qpos address):")
    for column, name in enumerate(expected):
        joint = model.joint(name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"mapped joint {name!r} is not a one-DoF hinge/slide joint")
        address = int(joint.qposadr[0])
        qpos_addresses.append(address)
        print(f"  {column:2d} {name:<32} -> qpos[{address}]")

    free_mapping: tuple[int, int] | None = None
    if free_ids:
        if len(free_ids) != 1:
            raise ValueError(f"expected at most one free joint, found {len(free_ids)}")
        free_id = free_ids[0]
        body_id = int(model.jnt_bodyid[free_id])
        body_name = object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        free_name = all_joint_names[free_id]
        free_qposadr = int(model.jnt_qposadr[free_id])
        print(f"free joint: {free_name}, body={body_name}, qpos address={free_qposadr}")
        if body_name != ROBONALDO_ROOT_BODY_NAME:
            raise ValueError(
                f"free joint belongs to {body_name!r}, but NPZ body index 0 is verified as "
                f"{ROBONALDO_ROOT_BODY_NAME!r}; use --fixed-root only after checking the model"
            )
        free_mapping = (free_id, free_qposadr)
    else:
        print("free joint: none (model root is fixed)")

    return model, qpos_addresses, free_mapping


def replay(
    model: mujoco.MjModel,
    arrays: dict[str, np.ndarray],
    fps: float,
    frames: int,
    qpos_addresses: list[int],
    free_mapping: tuple[int, int] | None,
    *,
    loop: bool,
    speed: float,
    fixed_root: bool,
) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    initial_qpos = data.qpos.copy()

    if fixed_root:
        print("root playback: fixed (--fixed-root)")
    elif free_mapping is None:
        print("root playback: model has no free joint; replaying joints only")
    else:
        print("root playback: body_pos_w[:, 0] + body_quat_w[:, 0] (wxyz)")

    period_wall = frames / (fps * speed)
    start = time.monotonic()
    last_frame = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            elapsed = time.monotonic() - start
            raw_frame = int(elapsed * fps * speed)
            if loop:
                frame = raw_frame % frames
                if elapsed >= period_wall:
                    # Avoid losing timer precision during very long loop sessions.
                    completed = int(elapsed / period_wall)
                    start += completed * period_wall
            else:
                frame = min(raw_frame, frames - 1)

            if frame != last_frame:
                data.qpos[:] = initial_qpos
                for column, address in enumerate(qpos_addresses):
                    data.qpos[address] = arrays["joint_pos"][frame, column]

                if not fixed_root and free_mapping is not None:
                    _, address = free_mapping
                    data.qpos[address : address + 3] = arrays["body_pos_w"][frame, 0]
                    data.qpos[address + 3 : address + 7] = arrays["body_quat_w"][frame, 0]

                mujoco.mj_forward(model, data)
                viewer.sync()
                last_frame = frame

            if not loop and raw_frame >= frames:
                break
            time.sleep(min(0.005, 0.25 / (fps * speed)))


def main() -> int:
    args = parse_args()
    try:
        arrays, fps, frames = inspect_npz(args.npz)
        model, qpos_addresses, free_mapping = inspect_model(args.model)
        if args.inspect_only:
            print("inspection passed; viewer not started (--inspect-only)")
            return 0
        replay(
            model,
            arrays,
            fps,
            frames,
            qpos_addresses,
            free_mapping,
            loop=args.loop,
            speed=args.speed,
            fixed_root=args.fixed_root,
        )
        return 0
    except (FileNotFoundError, OSError, ValueError, mujoco.FatalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

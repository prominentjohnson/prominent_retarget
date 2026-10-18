#!/usr/bin/env python3
"""Roll out RoboNaldo's kick tracking policy with real MuJoCo dynamics.

Unlike play_g1_npz.py, this script never overwrites the simulated robot pose
after initialization.  The robot moves only because MuJoCo integrates forces:

* policy: ONNX target positions -> PD torques -> MuJoCo actuators
* zero:   actuator torques are always zero
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the editor's Run button work on this Intel/NVIDIA hybrid laptop while
# still respecting any explicit environment chosen by the caller.
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

try:
    import mujoco
    import mujoco.viewer
    import onnxruntime as ort
    import yaml
except ImportError as exc:
    raise SystemExit(
        f"Missing Python dependency {exc.name!r}. Activate .venv and install "
        "onnxruntime and pyyaml manually."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = PROJECT_DIR / "third_party/RoboNaldo/RoboNaldo_Deploy"
DEFAULT_MODEL = DEPLOY_DIR / "g1_description/scene.xml"
DEFAULT_CONFIG = DEPLOY_DIR / "policy/beyondmimic/config/beyondmimic.yaml"
DEFAULT_MODE = "policy"

# MuJoCo DFS joint order used by the XML, actuators and policy config.
MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Official RoboNaldo mapping. For a vector in Isaac BFS order, indexing with
# this array produces MuJoCo DFS order. The inverse converts MuJoCo -> Isaac.
MUJOCO_TO_ISAAC = np.array(
    [
        0, 3, 6, 9, 13, 17,
        1, 4, 7, 10, 14, 18,
        2, 5, 8,
        11, 15, 19, 21, 23, 25, 27,
        12, 16, 20, 22, 24, 26, 28,
    ],
    dtype=np.int64,
)
ISAAC_TO_MUJOCO = np.argsort(MUJOCO_TO_ISAAC).astype(np.int64)
NPZ_TORSO_BODY_INDEX = 9
FALLEN_PELVIS_HEIGHT = 0.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo dynamics rollout of RoboNaldo's kick tracking policy."
    )
    parser.add_argument(
        "--mode",
        choices=("policy", "zero"),
        default=DEFAULT_MODE,
        help="policy applies ONNX+PD torque; zero applies exactly zero actuator torque",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="simulation seconds; 0 keeps running until the viewer is closed",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="wall-clock playback speed; does not change simulation/control dt",
    )
    parser.add_argument(
        "--ghost",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the kinematic reference as a translucent green robot",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without a viewer (requires a positive --duration)",
    )
    parser.add_argument("--report-interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.duration < 0.0 or not np.isfinite(args.duration):
        parser.error("--duration must be finite and >= 0")
    if args.headless and args.duration == 0.0:
        parser.error("--headless requires --duration > 0")
    if args.speed <= 0.0 or not np.isfinite(args.speed):
        parser.error("--speed must be positive and finite")
    if args.report_interval <= 0.0 or not np.isfinite(args.report_interval):
        parser.error("--report-interval must be positive and finite")
    return args


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, matrix.reshape(-1))
    return quat


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(result, np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64))
    return result


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def yaw_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def rot6d_from_quat(q: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q)[:, :2].reshape(-1).astype(np.float32)


class KickTrackingPolicy:
    """Minimal standalone reproduction of RoboNaldo_Deploy BeyondMimic."""

    def __init__(self, config_path: Path):
        if not config_path.is_file():
            raise FileNotFoundError(f"policy config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)

        policy_dir = config_path.parent.parent
        model_dir = policy_dir / "model"
        self.onnx_path = model_dir / cfg["onnx_path"]
        self.motion_path = model_dir / cfg["motion_path"]
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {self.onnx_path}")
        if not self.motion_path.is_file():
            raise FileNotFoundError(f"reference motion not found: {self.motion_path}")

        with np.load(self.motion_path, allow_pickle=False) as motion:
            self.joint_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
            self.joint_vel = np.asarray(motion["joint_vel"], dtype=np.float32)
            self.body_pos = np.asarray(motion["body_pos_w"], dtype=np.float32)
            self.body_quat = np.asarray(motion["body_quat_w"], dtype=np.float32)
            fps = float(np.asarray(motion["fps"]).reshape(-1)[0])

        self.control_dt = float(cfg["control_dt"])
        if not np.isclose(fps * self.control_dt, 1.0, atol=1e-6):
            raise ValueError(f"motion fps {fps} and control_dt {self.control_dt} disagree")
        if self.joint_pos.shape != self.joint_vel.shape or self.joint_pos.shape[1] != 29:
            raise ValueError("policy motion must contain matching [T, 29] joint_pos/joint_vel")
        if self.body_quat.shape[:2] != (self.joint_pos.shape[0], 30):
            raise ValueError("policy motion must contain body_quat_w with shape [T, 30, 4]")

        self.default_q_mj = np.asarray(cfg["default_joint_pos"], dtype=np.float32)
        self.default_q_il = self.default_q_mj[ISAAC_TO_MUJOCO]
        self.action_scale_mj = np.asarray(cfg["action_scale"], dtype=np.float32)
        self.kps = np.asarray(cfg["kps"], dtype=np.float64)
        self.kds = np.asarray(cfg["kds"], dtype=np.float64)
        self.tau_limit = np.asarray(cfg["tau_limit"], dtype=np.float64)
        self.clip_actions = float(cfg.get("clip_actions", 10.0))
        self.warmup_steps = int(cfg.get("warmup_steps", 10))

        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )
        inputs = {item.name: item.shape for item in self.session.get_inputs()}
        if inputs != {"obs": [1, 154], "time_step": [1, 1]}:
            raise ValueError(f"unexpected ONNX inputs: {inputs}")

        self.time_step = 0
        self.last_action_il = np.zeros(29, dtype=np.float32)
        self.init_to_world = np.eye(3, dtype=np.float64)
        self.entry_q_mj = np.zeros(29, dtype=np.float64)
        self.t0_target_mj = self.joint_pos[0][MUJOCO_TO_ISAAC].astype(np.float64)

    @property
    def frames(self) -> int:
        return self.joint_pos.shape[0]

    def reset(self, q_mj: np.ndarray, torso_quat_w: np.ndarray) -> None:
        self.time_step = 0
        self.last_action_il.fill(0.0)
        self.entry_q_mj = np.asarray(q_mj, dtype=np.float64).copy()
        motion_torso_q = self.body_quat[0, NPZ_TORSO_BODY_INDEX].astype(np.float64)
        relative_q = quat_mul(torso_quat_w, quat_conjugate(motion_torso_q))
        self.init_to_world = quat_to_matrix(yaw_quat(relative_q))

    def _observation(
        self,
        q_mj: np.ndarray,
        dq_mj: np.ndarray,
        torso_quat_w: np.ndarray,
        root_ang_vel_b: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        policy_step = self.time_step - self.warmup_steps
        frame = min(max(policy_step, 0), self.frames - 1)

        reference_torso_q = self.body_quat[frame, NPZ_TORSO_BODY_INDEX].astype(np.float64)
        aligned_q = quat_mul(matrix_to_quat(self.init_to_world), reference_torso_q)
        relative_q = quat_mul(quat_conjugate(torso_quat_w), aligned_q)
        relative_q /= np.linalg.norm(relative_q)

        q_il = np.asarray(q_mj)[ISAAC_TO_MUJOCO]
        dq_il = np.asarray(dq_mj)[ISAAC_TO_MUJOCO]
        obs = np.concatenate(
            [
                self.joint_pos[frame],
                self.joint_vel[frame],
                rot6d_from_quat(relative_q),
                np.asarray(root_ang_vel_b, dtype=np.float32),
                (q_il - self.default_q_il).astype(np.float32),
                dq_il.astype(np.float32),
                self.last_action_il,
            ],
            dtype=np.float32,
        )
        if obs.shape != (154,) or not np.isfinite(obs).all():
            raise ValueError("constructed policy observation is invalid")
        return obs, policy_step

    def target(
        self,
        q_mj: np.ndarray,
        dq_mj: np.ndarray,
        torso_quat_w: np.ndarray,
        root_ang_vel_b: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        if self.time_step < self.warmup_steps:
            alpha = (self.time_step + 1) / self.warmup_steps
            target_mj = (1.0 - alpha) * self.entry_q_mj + alpha * self.t0_target_mj
            frame = 0
        else:
            obs, policy_step = self._observation(
                q_mj, dq_mj, torso_quat_w, root_ang_vel_b
            )
            output = self.session.run(
                ["actions"],
                {"obs": obs[None, :], "time_step": np.array([[policy_step]], dtype=np.float32)},
            )[0][0]
            actions_il = np.clip(output, -self.clip_actions, self.clip_actions)
            self.last_action_il = actions_il.astype(np.float32, copy=True)
            actions_mj = actions_il[MUJOCO_TO_ISAAC]
            target_mj = self.default_q_mj + self.action_scale_mj * actions_mj
            frame = min(policy_step, self.frames - 1)

        self.time_step += 1
        return np.asarray(target_mj, dtype=np.float64), frame


def validate_model(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_joint_names = [model.joint(i).name for i in range(model.njnt)]
    missing = [name for name in MUJOCO_JOINT_NAMES if name not in model_joint_names]
    scalar = [name for name in model_joint_names if model.joint(name).type[0] != mujoco.mjtJoint.mjJNT_FREE]
    extra = [name for name in scalar if name not in MUJOCO_JOINT_NAMES]
    if missing or extra:
        raise ValueError(f"joint mismatch; missing={missing}, extra={extra}")

    qpos_addr = np.array([model.joint(name).qposadr[0] for name in MUJOCO_JOINT_NAMES], dtype=int)
    dof_addr = np.array([model.joint(name).dofadr[0] for name in MUJOCO_JOINT_NAMES], dtype=int)
    actuator_ids = []
    for name in MUJOCO_JOINT_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"model has no actuator named {name!r}")
        actuator_ids.append(actuator_id)
    return qpos_addr, dof_addr, np.asarray(actuator_ids, dtype=int)


def add_ghost_to_viewer(
    viewer,
    ghost_model: mujoco.MjModel,
    ghost_data: mujoco.MjData,
) -> None:
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        mujoco.mjv_addGeoms(
            ghost_model,
            ghost_data,
            mujoco.MjvOption(),
            mujoco.MjvPerturb(),
            mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
            viewer.user_scn,
        )


def run(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {args.model}")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    qpos_addr, dof_addr, actuator_ids = validate_model(model)
    policy = KickTrackingPolicy(args.config)

    ratio = policy.control_dt / model.opt.timestep
    control_decimation = int(round(ratio))
    if control_decimation < 1 or not np.isclose(ratio, control_decimation, atol=1e-9):
        raise ValueError(
            f"control_dt {policy.control_dt} is not an integer multiple of sim dt {model.opt.timestep}"
        )

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = policy.body_pos[0, 0]
    data.qpos[3:7] = policy.body_quat[0, 0]
    data.qpos[qpos_addr] = policy.joint_pos[0][MUJOCO_TO_ISAAC]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if torso_id < 0 or pelvis_id < 0:
        raise ValueError("model must contain pelvis and torso_link bodies")
    policy.reset(data.qpos[qpos_addr], data.xquat[torso_id])

    ghost_model = ghost_data = None
    if args.ghost and not args.headless:
        ghost_model = copy.deepcopy(model)
        ghost_model.geom_rgba[:, :3] = (0.2, 0.9, 0.4)
        ghost_model.geom_rgba[:, 3] = 0.22
        ghost_data = mujoco.MjData(ghost_model)

    print(f"mode: {args.mode}")
    print(f"model: {args.model.resolve()}")
    print(f"policy: {policy.onnx_path.resolve()}")
    print(f"reference: {policy.motion_path.resolve()} ({policy.frames} frames)")
    print(
        f"MuJoCo: {1 / model.opt.timestep:g} Hz; policy: {1 / policy.control_dt:g} Hz; "
        f"control decimation: {control_decimation}"
    )
    if args.mode == "zero":
        print("ZERO CONTROL ACTIVE: data.ctrl is held at exactly zero.")

    sim_steps = 0
    reference_frame = 0
    target_mj = data.qpos[qpos_addr].copy()
    min_pelvis_z = float(data.xpos[pelvis_id, 2])
    fallen = False
    next_report = 0.0
    wall_start = time.monotonic()
    render_interval = max(1, round((1.0 / 60.0) / model.opt.timestep))

    viewer_context = (
        _NullViewerContext() if args.headless else mujoco.viewer.launch_passive(model, data)
    )
    with viewer_context as viewer:
        while viewer.is_running():
            sim_time = sim_steps * model.opt.timestep
            if args.duration > 0.0 and sim_time >= args.duration:
                break

            if sim_steps % control_decimation == 0:
                if args.mode == "policy":
                    target_mj, reference_frame = policy.target(
                        data.qpos[qpos_addr],
                        data.qvel[dof_addr],
                        data.xquat[torso_id],
                        data.qvel[3:6],
                    )
                else:
                    reference_frame = min(
                        int(sim_time / policy.control_dt), policy.frames - 1
                    )

            if args.mode == "policy":
                torque_mj = (
                    policy.kps * (target_mj - data.qpos[qpos_addr])
                    - policy.kds * data.qvel[dof_addr]
                )
                torque_mj = np.clip(torque_mj, -policy.tau_limit, policy.tau_limit)
                data.ctrl[:] = 0.0
                data.ctrl[actuator_ids] = torque_mj
            else:
                data.ctrl[:] = 0.0

            mujoco.mj_step(model, data)
            sim_steps += 1
            sim_time = sim_steps * model.opt.timestep

            pelvis_z = float(data.xpos[pelvis_id, 2])
            min_pelvis_z = min(min_pelvis_z, pelvis_z)
            fallen = fallen or pelvis_z < FALLEN_PELVIS_HEIGHT

            if sim_time + 1e-12 >= next_report:
                q_reference_mj = policy.joint_pos[reference_frame][MUJOCO_TO_ISAAC]
                tracking_rmse = float(
                    np.sqrt(np.mean(np.square(data.qpos[qpos_addr] - q_reference_mj)))
                )
                max_torque = float(np.max(np.abs(data.ctrl))) if model.nu else 0.0
                print(
                    f"t={sim_time:6.2f}s pelvis_z={pelvis_z:.3f}m "
                    f"joint_rmse={tracking_rmse:.3f}rad max|tau|={max_torque:.1f}Nm "
                    f"fallen={'yes' if fallen else 'no'}"
                )
                next_report += args.report_interval

            if not args.headless and sim_steps % render_interval == 0:
                if ghost_model is not None and ghost_data is not None:
                    ghost_data.qpos[:3] = policy.body_pos[reference_frame, 0]
                    ghost_data.qpos[3:7] = policy.body_quat[reference_frame, 0]
                    ghost_data.qpos[qpos_addr] = policy.joint_pos[reference_frame][MUJOCO_TO_ISAAC]
                    mujoco.mj_forward(ghost_model, ghost_data)
                    add_ghost_to_viewer(viewer, ghost_model, ghost_data)
                viewer.sync()

            if not args.headless:
                target_wall = wall_start + sim_time / args.speed
                delay = target_wall - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)

    final_pelvis_z = float(data.xpos[pelvis_id, 2])
    print(
        f"result: mode={args.mode}, simulated={sim_steps * model.opt.timestep:.3f}s, "
        f"final_pelvis_z={final_pelvis_z:.3f}m, min_pelvis_z={min_pelvis_z:.3f}m, "
        f"fallen={'yes' if fallen else 'no'}"
    )
    return 0


class _NullViewerContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @staticmethod
    def is_running() -> bool:
        return True


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (FileNotFoundError, OSError, ValueError, RuntimeError, mujoco.FatalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""TwoBlockPick – minimal PyBullet environment with Franka Panda.

Observation (22-d):
    ee_pos(3)  ee_quat(4)  gripper(1)
    left_cube_pos(3)  left_cube_quat(4)
    right_cube_pos(3) right_cube_quat(4)

Action (5-d):
    dx  dy  dz  dyaw  grip   (each in [-1, 1])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

# ── dimensions ────────────────────────────────────────────────────────
OBS_DIM = 22
ACT_DIM = 5

# ── Panda constants ──────────────────────────────────────────────────
_EE_LINK = 11                    # panda_grasptarget
_ARM_JOINTS = list(range(7))     # revolute 0-6
_FINGER_JOINTS = [9, 10]         # prismatic
_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
_JOINT_LL = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
             0, 0, 0.0, 0.0, 0]
_JOINT_UL = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,
             0, 0, 0.04, 0.04, 0]
_JOINT_RANGES = [u - l for u, l in zip(_JOINT_UL, _JOINT_LL)]
_REST_POSES = _HOME_JOINTS + [0.0, 0.0, 0.04, 0.04, 0.0]

# ── scene layout ─────────────────────────────────────────────────────
_TABLE_POS   = [0.5, 0.0, 0.2]
_TABLE_HALF  = [0.30, 0.40, 0.20]   # top at z = 0.4
_TABLE_TOP_Z = 0.4

_CUBE_HALF   = 0.02                  # 4 cm cube
_CUBE_X      = 0.50
_CUBE_Y      = 0.07                  # ±0.07 left / right (symmetric about base centre line)
_CUBE_JITTER = 0.015
_CUBE_Z      = _TABLE_TOP_Z + _CUBE_HALF + 0.001

_SUCCESS_Z   = _TABLE_TOP_Z + 0.12   # "lifted" threshold (cube must be truly grasped + lifted)
_EE_HOME     = np.array([0.40, 0.0, 0.55], dtype=np.float32)

_SUBSTEPS = 20                       # physics steps per env step


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class TwoBlockPickEnv:
    """Minimal two-block pick environment (state-based, Franka Panda)."""

    def __init__(
        self,
        render: bool = False,
        action_scale_pos: float = 0.05,
        action_scale_yaw: float = math.radians(15),
        episode_length: int = 200,
        dt: float = 1.0 / 240.0,
        cube_jitter: float | None = None,
    ) -> None:
        self._renders = render
        self._action_scale_pos = action_scale_pos
        self._action_scale_yaw = action_scale_yaw
        self._episode_length = episode_length
        self._dt = dt
        self._cube_jitter = cube_jitter if cube_jitter is not None else _CUBE_JITTER

        # PyBullet connection
        mode = p.GUI if render else p.DIRECT
        self._cid = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._cid)

        self._rng = np.random.default_rng(0)
        self._episode_steps = 0
        self._picked_left = False
        self._picked_right = False

        # Video bookkeeping
        self._video_path: Optional[str] = None
        self._video_frames: List[np.ndarray] = []
        self._video_w = 640
        self._video_h = 480
        self._video_fps = 30

        # Body UIDs (set in _build_scene)
        self._plane_uid = -1
        self._table_uid = -1
        self._panda_uid = -1
        self._cube_l_uid = -1
        self._cube_r_uid = -1

        # EE tracking
        self._target_pos = _EE_HOME.copy()
        self._target_yaw = 0.0
        self._grip_cmd = 1.0  # +1 open, -1 closed

        self._build_scene()
        self.reset()

    # ── public API ────────────────────────────────────────────────────

    def close(self) -> None:
        if self._cid >= 0:
            p.disconnect(physicsClientId=self._cid)
            self._cid = -1

    def seed(self, s: int) -> None:
        self._rng = np.random.default_rng(s)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed(seed)

        self._reset_robot()
        self._reset_cubes()

        # settle physics
        for _ in range(120):
            p.stepSimulation(physicsClientId=self._cid)

        # sync target_pos with actual FK
        ee = p.getLinkState(self._panda_uid, _EE_LINK,
                            computeForwardKinematics=True,
                            physicsClientId=self._cid)
        self._target_pos = np.array(ee[4], dtype=np.float32)
        self._target_yaw = 0.0
        self._grip_cmd = 1.0

        self._episode_steps = 0
        self._picked_left = False
        self._picked_right = False
        self._video_frames = []
        return self._get_obs()

    def step(self, action: np.ndarray) -> StepResult:
        self._episode_steps += 1
        self._apply_action(np.asarray(action, dtype=np.float32))

        for _ in range(_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._cid)

        if self._video_path:
            self._capture()

        obs = self._get_obs()
        sl, sr = self._check_success()
        done = sl or sr or (self._episode_steps >= self._episode_length)
        reward = 1.0 if (sl or sr) else 0.0
        info = {
            "success_left": float(sl),
            "success_right": float(sr),
            "picked_left": float(self._picked_left),
            "picked_right": float(self._picked_right),
            "steps": self._episode_steps,
        }
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def render(self, mode: str = "rgb_array",
               width: int = 640, height: int = 480) -> np.ndarray:
        return self._cam_image(width, height)

    def record_video(self, path: str, width: int = 640,
                     height: int = 480, fps: int = 30) -> None:
        if self._video_path is not None:
            self._flush_video()
        self._video_path = path
        self._video_w = width
        self._video_h = height
        self._video_fps = fps
        self._video_frames = []
        self._capture()

    def stop_video(self) -> None:
        if self._video_path:
            self._flush_video()

    @property
    def episode_length(self) -> int:
        return self._episode_length

    @property
    def action_scale_pos(self) -> float:
        return self._action_scale_pos

    def set_cube_offsets(self,
                         left_dx: float = 0.0, left_dy: float = 0.0,
                         right_dx: float = 0.0, right_dy: float = 0.0,
                         ) -> None:
        """Reposition cubes with explicit offsets from default positions.

        Call AFTER reset() to override the jitter-based placement.
        Re-settles physics so the observation is stable.
        """
        ori = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(
            self._cube_l_uid,
            [_CUBE_X + left_dx, _CUBE_Y + left_dy, _CUBE_Z], ori,
            physicsClientId=self._cid)
        p.resetBaseVelocity(self._cube_l_uid, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._cid)
        p.resetBasePositionAndOrientation(
            self._cube_r_uid,
            [_CUBE_X + right_dx, -_CUBE_Y + right_dy, _CUBE_Z], ori,
            physicsClientId=self._cid)
        p.resetBaseVelocity(self._cube_r_uid, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._cid)
        # settle physics
        for _ in range(60):
            p.stepSimulation(physicsClientId=self._cid)

    # ── scene construction (called once) ─────────────────────────────

    def _build_scene(self) -> None:
        p.setGravity(0, 0, -9.81, physicsClientId=self._cid)
        p.setTimeStep(self._dt, physicsClientId=self._cid)
        p.setPhysicsEngineParameter(numSolverIterations=150,
                                     physicsClientId=self._cid)

        # ground plane
        self._plane_uid = p.loadURDF("plane.urdf", physicsClientId=self._cid)

        # table (static box)
        tc = p.createCollisionShape(p.GEOM_BOX, halfExtents=_TABLE_HALF,
                                    physicsClientId=self._cid)
        tv = p.createVisualShape(p.GEOM_BOX, halfExtents=_TABLE_HALF,
                                 rgbaColor=[0.6, 0.45, 0.25, 1],
                                 physicsClientId=self._cid)
        self._table_uid = p.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=tc,
                                            baseVisualShapeIndex=tv,
                                            basePosition=_TABLE_POS,
                                            physicsClientId=self._cid)

        # Panda robot
        self._panda_uid = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=self._cid,
        )
        # disable default velocity motors
        n_joints = p.getNumJoints(self._panda_uid, physicsClientId=self._cid)
        for j in range(n_joints):
            p.setJointMotorControl2(self._panda_uid, j, p.VELOCITY_CONTROL,
                                    force=0, physicsClientId=self._cid)
        # set initial arm joints via IK for EE_HOME
        self._reset_robot()

        # cubes (identical red)
        self._cube_l_uid = self._create_cube([1, 0.2, 0.2, 1])
        self._cube_r_uid = self._create_cube([1, 0.2, 0.2, 1])
        self._set_cube_friction(self._cube_l_uid)
        self._set_cube_friction(self._cube_r_uid)
        # finger friction
        for fj in _FINGER_JOINTS:
            p.changeDynamics(self._panda_uid, fj,
                             lateralFriction=1.5,
                             physicsClientId=self._cid)

    def _create_cube(self, rgba: list) -> int:
        cc = p.createCollisionShape(p.GEOM_BOX,
                                    halfExtents=[_CUBE_HALF] * 3,
                                    physicsClientId=self._cid)
        cv = p.createVisualShape(p.GEOM_BOX,
                                 halfExtents=[_CUBE_HALF] * 3,
                                 rgbaColor=rgba,
                                 physicsClientId=self._cid)
        uid = p.createMultiBody(baseMass=0.05,
                                baseCollisionShapeIndex=cc,
                                baseVisualShapeIndex=cv,
                                basePosition=[_CUBE_X, 0, _CUBE_Z],
                                physicsClientId=self._cid)
        return uid

    def _set_cube_friction(self, uid: int) -> None:
        p.changeDynamics(uid, -1,
                         lateralFriction=1.5,
                         spinningFriction=0.01,
                         rollingFriction=0.01,
                         physicsClientId=self._cid)

    # ── reset helpers ────────────────────────────────────────────────

    def _reset_robot(self) -> None:
        # Set approximate home, then refine with IK
        for i, q in enumerate(_HOME_JOINTS):
            p.resetJointState(self._panda_uid, i, q,
                              physicsClientId=self._cid)
        # open fingers (each finger to 0.02 = half of 0.04 total opening)
        for fj in _FINGER_JOINTS:
            p.resetJointState(self._panda_uid, fj, 0.02,
                              physicsClientId=self._cid)

        # IK for EE_HOME with gripper pointing down
        down_ori = p.getQuaternionFromEuler([math.pi, 0, 0])
        ik = p.calculateInverseKinematics(
            self._panda_uid, _EE_LINK,
            _EE_HOME.tolist(), down_ori,
            lowerLimits=_JOINT_LL, upperLimits=_JOINT_UL,
            jointRanges=_JOINT_RANGES, restPoses=_REST_POSES,
            maxNumIterations=300, residualThreshold=1e-5,
            physicsClientId=self._cid,
        )
        for i in _ARM_JOINTS:
            p.resetJointState(self._panda_uid, i, ik[i],
                              physicsClientId=self._cid)
            p.setJointMotorControl2(self._panda_uid, i,
                                    p.POSITION_CONTROL,
                                    targetPosition=ik[i],
                                    force=240,
                                    physicsClientId=self._cid)
        self._set_gripper(0.04)

    def _reset_cubes(self) -> None:
        jl = self._rng.uniform(-self._cube_jitter, self._cube_jitter)
        jr = self._rng.uniform(-self._cube_jitter, self._cube_jitter)
        jlx = self._rng.uniform(-self._cube_jitter, self._cube_jitter)
        jrx = self._rng.uniform(-self._cube_jitter, self._cube_jitter)
        ori = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(
            self._cube_l_uid,
            [_CUBE_X + jlx, _CUBE_Y + jl, _CUBE_Z], ori,
            physicsClientId=self._cid)
        p.resetBaseVelocity(self._cube_l_uid, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._cid)
        p.resetBasePositionAndOrientation(
            self._cube_r_uid,
            [_CUBE_X + jrx, -_CUBE_Y + jr, _CUBE_Z], ori,
            physicsClientId=self._cid)
        p.resetBaseVelocity(self._cube_r_uid, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._cid)

    # ── action processing ────────────────────────────────────────────

    def _apply_action(self, act: np.ndarray) -> None:
        assert act.shape == (ACT_DIM,), f"action shape {act.shape} != ({ACT_DIM},)"
        dx, dy, dz, dyaw, grip = act

        self._target_pos += np.array([dx, dy, dz]) * self._action_scale_pos
        self._target_pos = np.clip(self._target_pos,
                                   [0.25, -0.30, 0.05],
                                   [0.70,  0.30, 0.70])

        self._target_yaw += float(dyaw) * self._action_scale_yaw
        self._target_yaw = float(np.clip(self._target_yaw, -math.pi, math.pi))

        target_ori = p.getQuaternionFromEuler([math.pi, 0, self._target_yaw])

        ik = p.calculateInverseKinematics(
            self._panda_uid, _EE_LINK,
            self._target_pos.tolist(), target_ori,
            lowerLimits=_JOINT_LL, upperLimits=_JOINT_UL,
            jointRanges=_JOINT_RANGES, restPoses=_REST_POSES,
            maxNumIterations=100, residualThreshold=1e-4,
            physicsClientId=self._cid,
        )
        for i in _ARM_JOINTS:
            p.setJointMotorControl2(self._panda_uid, i,
                                    p.POSITION_CONTROL,
                                    targetPosition=ik[i],
                                    force=240,
                                    physicsClientId=self._cid)

        self._grip_cmd = float(np.clip(grip, -1, 1))
        opening = 0.04 if self._grip_cmd > 0 else 0.001
        self._set_gripper(opening)

    def _set_gripper(self, opening: float) -> None:
        finger_pos = opening / 2.0
        for fj in _FINGER_JOINTS:
            p.setJointMotorControl2(
                self._panda_uid, fj, p.POSITION_CONTROL,
                targetPosition=finger_pos, force=40,
                physicsClientId=self._cid)

    # ── observation ──────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        ee = p.getLinkState(self._panda_uid, _EE_LINK,
                            computeForwardKinematics=True,
                            physicsClientId=self._cid)
        ee_pos = np.array(ee[4], dtype=np.float32)          # 3
        ee_quat = np.array(ee[5], dtype=np.float32)         # 4
        grip = np.array([self._grip_cmd], dtype=np.float32)  # 1

        lp, lo = p.getBasePositionAndOrientation(self._cube_l_uid,
                                                  physicsClientId=self._cid)
        rp, ro = p.getBasePositionAndOrientation(self._cube_r_uid,
                                                  physicsClientId=self._cid)

        return np.concatenate([
            ee_pos, ee_quat, grip,
            np.array(lp, dtype=np.float32), np.array(lo, dtype=np.float32),
            np.array(rp, dtype=np.float32), np.array(ro, dtype=np.float32),
        ])  # 22

    # ── success check ────────────────────────────────────────────────

    def _check_success(self) -> Tuple[bool, bool]:
        lz = p.getBasePositionAndOrientation(
            self._cube_l_uid, physicsClientId=self._cid)[0][2]
        rz = p.getBasePositionAndOrientation(
            self._cube_r_uid, physicsClientId=self._cid)[0][2]

        sl = lz > _SUCCESS_Z
        sr = rz > _SUCCESS_Z

        if sl and not self._picked_left and not self._picked_right:
            self._picked_left = True
        if sr and not self._picked_right and not self._picked_left:
            self._picked_right = True

        return sl, sr

    # ── camera / video ───────────────────────────────────────────────

    def _cam_image(self, w: int, h: int) -> np.ndarray:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.50, 0.0, _TABLE_TOP_Z],
            distance=0.9, yaw=135, pitch=-30, roll=0, upAxisIndex=2)
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=w / h, nearVal=0.1, farVal=3.0)
        renderer = (p.ER_BULLET_HARDWARE_OPENGL if self._renders
                     else p.ER_TINY_RENDERER)
        _, _, px, _, _ = p.getCameraImage(w, h, view, proj,
                                          renderer=renderer,
                                          physicsClientId=self._cid)
        img = np.array(px, dtype=np.uint8).reshape(h, w, 4)
        return img[:, :, :3]

    def _capture(self) -> None:
        self._video_frames.append(
            self._cam_image(self._video_w, self._video_h))

    def _flush_video(self) -> None:
        if not self._video_frames or not self._video_path:
            self._video_path = None
            self._video_frames = []
            return
        path = self._video_path
        frames = self._video_frames
        self._video_path = None
        self._video_frames = []
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            import imageio.v2 as iio
            iio.mimsave(path, frames, fps=self._video_fps,
                        macro_block_size=None)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] video save failed: {exc}")


# ── factory ──────────────────────────────────────────────────────────

def make_env(render: bool = False, **kw: Any) -> TwoBlockPickEnv:
    return TwoBlockPickEnv(render=render, **kw)

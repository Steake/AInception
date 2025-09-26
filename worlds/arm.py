"""
2-DOF Planar Arm Environment (V1)

Objective:
  - Move end-effector to target while avoiding forbidden zone and optionally distractor targets.
  - Models simple kinematics only (no dynamics integration for MVP).
  - Provides stochastic perturbations (angle noise / slip) to encourage robustness.

State Variables:
  joint_angles: (theta1, theta2) in radians
  target_pos: (x,y)
  distractors: list[(x,y)]
  forbidden_zone: axis-aligned bounding box ((x_min,y_min),(x_max,y_max))
  reached: bool
  tick: int

Actions (two modes supported):
  1. Discrete torque-ish increments:
     {"type":"inc", "j1": d_theta1, "j2": d_theta2}
     where d_theta values chosen from allowed step set (e.g. [-step, 0, step])
  2. No-op:
     {"type":"noop"}

Kinematics:
  link lengths: L1, L2
  End effector position:
     x = L1*cos(t1) + L2*cos(t1 + t2)
     y = L1*sin(t1) + L2*sin(t1 + t2)

Reward (optional shaping):
  r = - distance_to_target
      - forbidden_penalty (if inside)
      - step_cost
    + success_bonus (once when within success_radius)

Forbidden Handling:
  - Entering forbidden zone triggers high penalty and flags potential principle violation externally.

Noise:
  - After applying action increments, Gaussian noise with std angle_noise applied per joint (clamped).

Termination:
  - success (reached target)
  - max_ticks exceeded

Observation:
  {
    "joint_angles": (t1, t2),
    "end_effector": (ex, ey),
    "target": (tx, ty),
    "distractors": [(..), ...],
    "distance_to_target": float,
    "forbidden_zone": ((x0,y0),(x1,y1)),
    "in_forbidden": bool,
    "tick": int
  }

Integration Notes:
  - Agent can treat (distance_to_target, in_forbidden) as additional features for drives or risk.
  - Promise system may register a promise to avoid forbidden zone center, enforced externally.

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import math
import random


@dataclass
class ArmConfig:
    link_lengths: Tuple[float, float] = (1.0, 1.0)
    initial_angles: Tuple[float, float] = (0.0, 0.0)
    target_pos: Optional[Tuple[float, float]] = None
    random_target_radius: float = 1.6
    distractor_count: int = 2
    distractor_radius: float = 1.6
    forbidden_zone: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.5, 0.5), (0.8, 0.8))
    success_radius: float = 0.08
    max_ticks: int = 300
    step_angle: float = 0.12
    angle_noise_std: float = 0.01
    joint_limits: Tuple[Tuple[float, float], Tuple[float, float]] = ((-math.pi, math.pi), (-math.pi, math.pi))
    step_cost: float = 0.01
    forbidden_penalty: float = 0.5
    success_bonus: float = 1.0
    seed: Optional[int] = None
    use_continuous: bool = False  # (future extension; currently discrete increments)
    allow_target_reset_on_success: bool = False


class ArmEnv:
    def __init__(self, config: ArmConfig | None = None):
        self.cfg = config or ArmConfig()
        self.rng = random.Random(self.cfg.seed)

        # State
        self.joint_angles: List[float] = list(self.cfg.initial_angles)
        self.target_pos: Tuple[float, float] = self.cfg.target_pos or self._sample_target(self.cfg.random_target_radius)
        self.distractors: List[Tuple[float, float]] = []
        self.reached: bool = False
        self.tick: int = 0
        self.done: bool = False

        self._init_distractors()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
        self.tick = 0
        self.done = False
        self.reached = False
        self.joint_angles = list(self.cfg.initial_angles)
        if self.cfg.target_pos is None:
            self.target_pos = self._sample_target(self.cfg.random_target_radius)
        self._init_distractors()
        return self._observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self._observation(), 0.0, True, {"already_done": True}

        self.tick += 1
        info: Dict[str, Any] = {}
        reward = 0.0

        a_type = action.get("type", "noop")
        if a_type == "inc":
            d1 = float(action.get("j1", 0.0))
            d2 = float(action.get("j2", 0.0))
            self._apply_increment(d1, d2)
            reward -= self.cfg.step_cost
        elif a_type == "noop":
            reward -= 0.5 * self.cfg.step_cost
        else:
            # Unknown action type; treat as noop
            reward -= 0.5 * self.cfg.step_cost

        ee = self._end_effector()
        dist = self._distance(ee, self.target_pos)
        reward -= dist

        in_forbidden = self._in_forbidden(ee)
        if in_forbidden:
            reward -= self.cfg.forbidden_penalty
            info["forbidden_contact"] = True

        if dist <= self.cfg.success_radius:
            if not self.reached:
                self.reached = True
                reward += self.cfg.success_bonus
                info["success"] = True
                if self.cfg.allow_target_reset_on_success:
                    # Reset target to new random location
                    self.target_pos = self._sample_target(self.cfg.random_target_radius)
                    self.reached = False  # continuing curriculum
            else:
                # Already reached previously; small shaping
                reward += 0.1

        if self.tick >= self.cfg.max_ticks:
            self.done = True
            info["termination"] = "max_ticks"
        elif self.reached and not self.cfg.allow_target_reset_on_success:
            self.done = True
            info["termination"] = "success"

        obs = self._observation()
        return obs, reward, self.done, info

    # ------------------------------------------------------------------
    # Internal Mechanics
    # ------------------------------------------------------------------
    def _apply_increment(self, d1: float, d2: float):
        # Clamp increments to allowed step multiples
        step = self.cfg.step_angle
        d1 = self._snap(d1, step)
        d2 = self._snap(d2, step)
        self.joint_angles[0] += d1
        self.joint_angles[1] += d2
        # Add noise
        self.joint_angles[0] += self.rng.gauss(0, self.cfg.angle_noise_std)
        self.joint_angles[1] += self.rng.gauss(0, self.cfg.angle_noise_std)
        # Clamp within joint limits
        (l1_min, l1_max), (l2_min, l2_max) = self.cfg.joint_limits
        self.joint_angles[0] = max(l1_min, min(l1_max, self.joint_angles[0]))
        self.joint_angles[1] = max(l2_min, min(l2_max, self.joint_angles[1]))

    def _snap(self, val: float, step: float) -> float:
        # Snap to nearest step multiple (including zero)
        m = round(val / step)
        return m * step

    def _end_effector(self) -> Tuple[float, float]:
        L1, L2 = self.cfg.link_lengths
        t1, t2 = self.joint_angles
        x = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
        y = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
        return (x, y)

    def _in_forbidden(self, pt: Tuple[float, float]) -> bool:
        (x0, y0), (x1, y1) = self.cfg.forbidden_zone
        x, y = pt
        return x0 <= x <= x1 and y0 <= y <= y1

    def _sample_target(self, radius: float) -> Tuple[float, float]:
        # Sample uniformly within circle of given radius centered at origin
        for _ in range(200):
            r = radius * math.sqrt(self.rng.random())
            angle = 2 * math.pi * self.rng.random()
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            if not self._in_forbidden((x, y)):
                return (x, y)
        return (radius * 0.5, radius * 0.5)

    def _init_distractors(self):
        self.distractors = []
        for _ in range(self.cfg.distractor_count):
            self.distractors.append(self._sample_target(self.cfg.distractor_radius))

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.dist(a, b)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _observation(self) -> Dict[str, Any]:
        ee = self._end_effector()
        return {
            "joint_angles": tuple(self.joint_angles),
            "end_effector": ee,
            "target": self.target_pos,
            "distractors": list(self.distractors),
            "distance_to_target": self._distance(ee, self.target_pos),
            "forbidden_zone": self.cfg.forbidden_zone,
            "in_forbidden": self._in_forbidden(ee),
            "tick": self.tick
        }

    # ------------------------------------------------------------------
    # Rendering / Debug
    # ------------------------------------------------------------------
    def render_ascii(self, scale: int = 20) -> str:
        """
        Render coarse ASCII plane around origin within range [-2,2] for each axis.
        This is approximate and mainly for debugging.
        """
        size = scale
        half = size // 2
        canvas = [["." for _ in range(size)] for _ in range(size)]

        def to_grid(pt: Tuple[float, float]):
            # map x,y in approx [-2,2] to grid
            x, y = pt
            gx = int((x / 2.0) * half + half)
            gy = int((y / 2.0) * half + half)
            return gx, (size - 1 - gy)

        # Mark forbidden zone corners
        (fx0, fy0), (fx1, fy1) = self.cfg.forbidden_zone
        corners = [(fx0, fy0), (fx0, fy1), (fx1, fy0), (fx1, fy1)]
        for c in corners:
            gx, gy = to_grid(c)
            if 0 <= gx < size and 0 <= gy < size:
                canvas[gy][gx] = "F"

        # Mark target
        gx, gy = to_grid(self.target_pos)
        if 0 <= gx < size and 0 <= gy < size:
            canvas[gy][gx] = "T"

        # Mark distractors
        for d in self.distractors:
            gx, gy = to_grid(d)
            if 0 <= gx < size and 0 <= gy < size:
                canvas[gy][gx] = "D"

        # Mark end effector
        ee = self._end_effector()
        gx, gy = to_grid(ee)
        if 0 <= gx < size and 0 <= gy < size:
            canvas[gy][gx] = "E"

        return "\n".join("".join(row) for row in canvas)


# ------------------------------------------------------------------
# Self Test
# ------------------------------------------------------------------
if __name__ == "__main__":
    cfg = ArmConfig(seed=3)
    env = ArmEnv(cfg)
    obs = env.reset()
    print("Initial:", obs)
    for i in range(10):
        a = {"type": "inc", "j1": 0.12, "j2": 0.0}
        obs, r, done, info = env.step(a)
        print(f"Step {i} r={r:.3f} done={done} info={info}")
        if done:
            break
    print("Render:\n", env.render_ascii())
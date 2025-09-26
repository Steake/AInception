"""
Grid World Environment (V1)

Goals:
- Maintain drives (energy, temperature) while delivering an item from spawn to target
- Avoid danger / forbidden tiles (principle & promise compliance)
- Provide stochastic dynamics (slip / temperature noise)
- Offer resource tiles that replenish energy
- Provide warm zones affecting temperature

State Variables:
  agent_pos: (x,y)
  item_pos: (x,y) or None if delivered or being carried
  target_pos: (x,y)
  carrying: bool
  energy: float in [0,1]
  temperature: float in [0,1]
  social_proximity: float (stub; could be distance to hypothetical peer)
  tick: int

Actions:
  {"type": "move", "dx": ±1|0, "dy": ±1|0} (cardinal only)
  {"type": "pick"}  (pick up item if on same cell and not carrying)
  {"type": "drop"}  (drop item; if on target cell => delivery success)
  {"type": "noop"}

Stochasticity:
  - Slip chance: with probability slip_chance, movement deviates perpendicular if available
  - Temperature noise: additive Gaussian clipped
  - Energy decay each step + move cost

Reward (optional usage):
  - Provided as shaped info; not strictly required by agent which uses drives
  - r = delivery_bonus (once) - step_cost - danger_penalty(if on danger tile)
    (Homeostatic drives handled outside environment in agent logic)

Observation Dict Keys (aligned with perception / drives):
  {
    "agent_pos": (x,y),
    "goal": target_pos,
    "energy": energy,
    "temperature": temperature,
    "social_proximity": social_proximity,
    "danger_tiles": set(...),
    "forbidden_tiles": set(...),
    "tick": tick
  }

Promises:
  - External harness may register promise to avoid specific tile(s). We expose forbidden_tiles for the planner filter.

Termination:
  - Episode ends if:
      * energy <= 0
      * max_ticks reached
      * item delivered (success flag success=True but can continue if allow_continued=False)

Usage:
  env = GridWorld(width=15, height=15, seed=42)
  obs = env.reset()
  obs, reward, done, info = env.step({"type":"move","dx":1,"dy":0})

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set
import random
import math


@dataclass
class GridWorldConfig:
    width: int = 15
    height: int = 15
    warm_zones: List[Tuple[int, int]] = field(default_factory=list)
    danger_tiles: List[Tuple[int, int]] = field(default_factory=list)
    forbidden_tiles: List[Tuple[int, int]] = field(default_factory=list)  # used for promises
    energy_sources: List[Tuple[int, int]] = field(default_factory=list)
    initial_energy: float = 0.7
    energy_decay: float = 0.01
    move_energy_cost: float = 0.01
    pick_energy_cost: float = 0.005
    drop_energy_cost: float = 0.005
    energy_gain_on_source: float = 0.15
    temperature_base: float = 0.5
    warm_zone_bonus: float = 0.15
    temperature_noise_std: float = 0.02
    slip_chance: float = 0.05
    max_ticks: int = 400
    delivery_bonus: float = 1.0
    danger_penalty: float = 0.2
    allow_continued_after_delivery: bool = False
    random_item: bool = True
    random_target: bool = True


class GridWorld:
    def __init__(
        self, 
        config: GridWorldConfig | None = None, 
        seed: Optional[int] = None,
        # Alternative constructor parameters for CLI compatibility
        width: Optional[int] = None,
        height: Optional[int] = None,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        danger_tiles: Optional[Set[Tuple[int, int]]] = None,
        forbidden_tiles: Optional[Set[Tuple[int, int]]] = None
    ):
        # If individual parameters provided, create config from them
        if any(x is not None for x in [width, height, start_pos, goal_pos, danger_tiles, forbidden_tiles]):
            self.cfg = GridWorldConfig(
                width=width or 10,
                height=height or 10,
                danger_tiles=list(danger_tiles) if danger_tiles else [],
                forbidden_tiles=list(forbidden_tiles) if forbidden_tiles else []
            )
            # Override start and goal positions
            self._start_pos = start_pos or (0, 0)
            self._goal_pos = goal_pos or (width-1 if width else 9, height-1 if height else 9)
        else:
            self.cfg = config or GridWorldConfig()
            self._start_pos = (0, 0)
            self._goal_pos = (self.cfg.width - 2, self.cfg.height - 2)
            
        self.rng = random.Random(seed)
        self.width = self.cfg.width
        self.height = self.cfg.height

        # Dynamic episode state
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.item_pos: Optional[Tuple[int, int]] = None
        self.target_pos: Tuple[int, int] = (self.width - 2, self.height - 2)
        self.carrying: bool = False
        self.energy: float = self.cfg.initial_energy
        self.temperature: float = self.cfg.temperature_base
        self.social_proximity: float = 0.2  # stub; could be driven by a peer agent distance
        self.tick: int = 0
        self.delivered: bool = False
        self.done: bool = False

        # Cached sets
        self._warm_zones: Set[Tuple[int, int]] = set(self.cfg.warm_zones)
        self._danger_tiles: Set[Tuple[int, int]] = set(self.cfg.danger_tiles)
        self._forbidden_tiles: Set[Tuple[int, int]] = set(self.cfg.forbidden_tiles)
        self._energy_sources: Set[Tuple[int, int]] = set(self.cfg.energy_sources)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)

        self.tick = 0
        self.done = False
        self.delivered = False
        self.carrying = False
        self.energy = self.cfg.initial_energy
        self.temperature = self.cfg.temperature_base

        # Set agent start position
        self.agent_pos = self._start_pos

        # Item position
        if self.cfg.random_item:
            self.item_pos = self._sample_free_tile(avoid={self.agent_pos})
        else:
            self.item_pos = (1, 1)

        # Target position - use custom goal if provided
        self.target_pos = self._goal_pos

        return self._observation()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step method compatible with CLI framework.
        Returns dict with observation, reward, done, info.
        """
        obs, reward, done, info = self._step_internal(action)
        return {
            "observation": obs,
            "reward": reward, 
            "done": done,
            "info": info
        }

    def _step_internal(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self._observation(), 0.0, True, {"already_done": True}

        self.tick += 1
        reward = 0.0
        info: Dict[str, Any] = {}

        at_before = self.agent_pos

        # Apply action
        a_type = action.get("type", "noop")
        if a_type == "move":
            dx = int(action.get("dx", 0))
            dy = int(action.get("dy", 0))
            self._apply_move(dx, dy)
            self.energy -= self.cfg.move_energy_cost
        elif a_type == "pick":
            reward -= self.cfg.pick_energy_cost
            self.energy -= self.cfg.pick_energy_cost
            if not self.carrying and self.item_pos == self.agent_pos and self.item_pos is not None:
                self.carrying = True
                self.item_pos = None
                info["picked"] = True
        elif a_type == "drop":
            reward -= self.cfg.drop_energy_cost
            self.energy -= self.cfg.drop_energy_cost
            if self.carrying:
                if self.agent_pos == self.target_pos:
                    self.delivered = True
                    self.carrying = False
                    info["delivered"] = True
                    reward += self.cfg.delivery_bonus
                else:
                    # Drop item on ground
                    self.item_pos = self.agent_pos
                    self.carrying = False
                    info["dropped"] = True
        else:  # noop
            pass

        # Environment stochastic influences
        self._update_temperature()
        self._maybe_energy_collect()

        # Passive decay
        self.energy -= self.cfg.energy_decay
        if self.energy < 0:
            self.energy = 0.0

        # Danger penalty
        if self.agent_pos in self._danger_tiles:
            reward -= self.cfg.danger_penalty
            info["danger_contact"] = True

        # Termination checks
        if self.energy <= 0:
            self.done = True
            info["termination"] = "energy_depleted"
        elif self.tick >= self.cfg.max_ticks:
            self.done = True
            info["termination"] = "max_ticks"
        elif self.delivered and not self.cfg.allow_continued_after_delivery:
            self.done = True
            info["termination"] = "delivered"

        obs = self._observation()
        return obs, reward, self.done, info

    # ---------------------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------------------
    def _apply_move(self, dx: int, dy: int):
        # Enforce cardinal
        if abs(dx) + abs(dy) != 1:
            dx, dy = 0, 0

        # Slip?
        if self.rng.random() < self.cfg.slip_chance:
            # Perpendicular slip if possible
            if dx != 0:
                dx, dy = 0, self.rng.choice([-1, 1])
            elif dy != 0:
                dx, dy = self.rng.choice([-1, 1]), 0

        nx = max(0, min(self.width - 1, self.agent_pos[0] + dx))
        ny = max(0, min(self.height - 1, self.agent_pos[1] + dy))
        self.agent_pos = (nx, ny)

    def _update_temperature(self):
        # Base
        t = self.cfg.temperature_base
        # Add warm zone effect if standing on warm tile
        if self.agent_pos in self._warm_zones:
            t += self.cfg.warm_zone_bonus
        # Add mild gradient relative to center (simulate environment variety)
        cx, cy = (self.width - 1) / 2.0, (self.height - 1) / 2.0
        ax, ay = self.agent_pos
        dist_center = math.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
        t -= 0.02 * (dist_center / max(self.width, self.height))
        # Noise
        noise = self.rng.gauss(0, self.cfg.temperature_noise_std)
        t += noise
        # Clamp [0,1]
        self.temperature = max(0.0, min(1.0, t))

    def _maybe_energy_collect(self):
        if self.agent_pos in self._energy_sources:
            self.energy = min(1.0, self.energy + self.cfg.energy_gain_on_source)

    def _sample_free_tile(self, avoid: Set[Tuple[int, int]]) -> Tuple[int, int]:
        # Avoid danger / forbidden for spawn to simplify
        blacklist = set(avoid) | self._danger_tiles | self._forbidden_tiles
        for _ in range(500):
            x = self.rng.randint(0, self.width - 1)
            y = self.rng.randint(0, self.height - 1)
            if (x, y) not in blacklist:
                return (x, y)
        # Fallback: origin
        return (0, 0)

    # ---------------------------------------------------------------------
    # Observation Assembly
    # ---------------------------------------------------------------------
    def _observation(self) -> Dict[str, Any]:
        return {
            "agent_pos": self.agent_pos,
            "goal": self.target_pos,
            "energy": self.energy,
            "temperature": self.temperature,
            "social_proximity": self.social_proximity,
            "danger_tiles": set(self._danger_tiles),
            "forbidden_tiles": set(self._forbidden_tiles),
            "carrying": self.carrying,
            "item_pos": self.item_pos,
            "delivered": self.delivered,
            "tick": self.tick
        }

    # ---------------------------------------------------------------------
    # Introspection / Utility
    # ---------------------------------------------------------------------
    def render_ascii(self) -> str:
        """
        Returns an ASCII grid depiction for debugging.
        Legend:
          A = Agent
          I = Item (not carried)
          T = Target
          D = Danger
          F = Forbidden
          W = Warm
          E = Energy source
        """
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for (x, y) in self._danger_tiles:
            if self._in_bounds(x, y):
                grid[y][x] = "D"
        for (x, y) in self._forbidden_tiles:
            if self._in_bounds(x, y):
                grid[y][x] = "F"
        for (x, y) in self._warm_zones:
            if self._in_bounds(x, y):
                grid[y][x] = "W"
        for (x, y) in self._energy_sources:
            if self._in_bounds(x, y):
                grid[y][x] = "E"
        if self.item_pos:
            ix, iy = self.item_pos
            if self._in_bounds(ix, iy):
                grid[iy][ix] = "I"
        tx, ty = self.target_pos
        if self._in_bounds(tx, ty):
            grid[ty][tx] = "T"
        ax, ay = self.agent_pos
        if self._in_bounds(ax, ay):
            grid[ay][ax] = "A"

        return "\n".join("".join(row) for row in grid)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def check_goal_reached(self, observation: Dict[str, Any]) -> bool:
        """Check if the goal has been reached (item delivered)."""
        return self.delivered

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation (public interface)."""
        return self._observation()


# -------------------------------------------------------------------------
# Basic Self-Test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = GridWorldConfig(
        width=10,
        height=8,
        warm_zones=[(2, 2), (2, 3)],
        danger_tiles=[(5, 5)],
        forbidden_tiles=[(6, 6)],
        energy_sources=[(1, 1), (7, 2)]
    )
    env = GridWorld(cfg, seed=7)
    obs = env.reset()
    print("Initial Observation:", obs)
    for i in range(5):
        a = {"type": "move", "dx": 1, "dy": 0}
        obs, r, done, info = env.step(a)
        print(f"Step {i} action={a} reward={r:.3f} done={done} info={info}")
        print(env.render_ascii())
        if done:
            break
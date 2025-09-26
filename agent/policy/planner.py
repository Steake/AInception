"""
Deliberative Planner: Hybrid A* + Constrained MPC (skeleton implementation)
Includes:
 - Candidate generation via bounded A* (with early pruning)
 - Optional neural augmentation hooks (spatial-temporal embeddings + predictions)
 - Cost evaluation J = alpha*steps + beta*drive_cost + gamma*risk + delta*energy + lambda*promise_pen + mu*principle_pen
 - Justification object assembly
 - Surprise prediction cache hooks
 - Fallback logic when no safe candidate survives
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable
import heapq
import math
import random
import time


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CostWeights:
    alpha: float = 1.0          # step distance / path length
    beta: float = 1.0           # drive deviation
    gamma: float = 1.0          # risk
    delta: float = 0.2          # energy use
    lambd: float = 5.0          # promise penalty
    mu: float = 5.0             # principle penalty


@dataclass
class CandidatePath:
    nodes: List[Any]
    actions: List[Any]
    heuristic: float
    g_cost: float
    predicted_drive_cost: float = 0.0
    predicted_risk_cost: float = 0.0
    predicted_energy_delta: float = 0.0
    promise_violation: bool = False
    principle_violation: bool = False
    discarded_reason: Optional[str] = None

    def length(self) -> int:
        return len(self.actions)

    def total_steps(self) -> int:
        return len(self.nodes) - 1 if self.nodes else 0


@dataclass
class NeuralPredictions:
    drive_cost: float
    risk_score: float
    energy_delta: float
    confidence: float


@dataclass
class PlannerConfig:
    candidate_k: int = 6
    prefix_h: int = 6
    imagination_horizons: Tuple[int, ...] = (1, 3)
    max_expansions: int = 500
    allow_diagonal: bool = False
    surprise_threshold: float = 0.25
    consecutive_surprise_limit: int = 3
    confidence_min: float = 0.55
    risk_cap_for_heuristic: float = 0.8
    use_neural: bool = True


@dataclass
class Justification:
    chosen_goal: Optional[str]
    principles_checked: List[str]
    rejected_candidates_count: int
    dominant_drive: Optional[str]
    sacrificed_metric: Optional[str]
    promise_status: str
    model_confidence: Optional[float]
    model_used: bool
    heuristic_augmented: bool
    fallback_reason: Optional[str]
    cost_breakdown: Dict[str, float]
    surprise_flag: bool = False
    risk_inflation_applied: bool = False
    override_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Interfaces / Hooks
# ---------------------------------------------------------------------------

class NeuralModuleInterface:
    """
    Expected interface for neural augmentation module.
    Actual implementation injected from outside.
    """
    def predict_state(self, state: Dict[str, Any]) -> NeuralPredictions:
        return NeuralPredictions(
            drive_cost=0.0,
            risk_score=0.0,
            energy_delta=0.0,
            confidence=0.0
        )

    def predict_path(self, state: Dict[str, Any], path: CandidatePath) -> NeuralPredictions:
        # Optional path-level refinement
        return self.predict_state(state)


# ---------------------------------------------------------------------------
# Planner Implementation
# ---------------------------------------------------------------------------

class Planner:
    def __init__(
        self,
        config: PlannerConfig,
        cost_weights: CostWeights,
        get_neighbors: Callable[[Any], List[Tuple[Any, Any]]],
        is_principle_violation: Callable[[Any], bool],
        is_promise_violation: Callable[[Any], bool],
        drive_error_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        goal_fn: Callable[[Dict[str, Any]], Optional[Any]],
        state_position_extractor: Callable[[Dict[str, Any]], Any],
        neural_module: Optional[NeuralModuleInterface] = None,
        promise_book: Optional[Any] = None,
        imagination: Optional[Any] = None,
        drive_system: Optional[Any] = None,
        rng: Optional[random.Random] = None
    ):
        """
        get_neighbors(node) -> List of (neighbor_node, action)
        is_principle_violation(node) -> bool
        is_promise_violation(node) -> bool
        drive_error_fn(state_dict) -> drive errors {drive: error_value}
        goal_fn(state_dict) -> goal position / target state abstract
        state_position_extractor(state_dict) -> canonical node position
        """
        self.cfg = config
        self.weights = cost_weights
        self.get_neighbors = get_neighbors
        self.is_principle_violation = is_principle_violation
        self.is_promise_violation = is_promise_violation
        self.drive_error_fn = drive_error_fn
        self.goal_fn = goal_fn
        self.extract_pos = state_position_extractor
        self.neural = neural_module
        self.promise_book = promise_book
        self.imagination = imagination
        self.drive_system = drive_system
        self.rng = rng or random.Random(0)

        # Internal tracking for surprise / confidence adaptation
        self.region_surprise_counts: Dict[str, int] = {}

    # ------------------------------- Public API -----------------------------

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Top-level planning call.
        Returns dict with action + justification skeleton.
        """
        start_node = self.extract_pos(state)
        goal = self.goal_fn(state)
        if goal is None:
            return self._noop_justified("no_goal")

        # Stage 1: generate candidate prefixes by bounded A*
        candidates, rejected_count, augmented = self._generate_candidates(state, start_node, goal)

        # Stage 2: evaluate with constrained MPC style scoring
        scored = self._score_candidates(state, candidates)

        # Select best or fallback
        chosen, fallback_reason = self._select_best(scored)

        justification = self._build_justification(
            state=state,
            chosen=chosen,
            rejected=rejected_count,
            augmented=augmented,
            fallback_reason=fallback_reason
        )

        action = {"type": "noop"} if chosen is None else chosen.actions[0] if chosen.actions else {"type": "noop"}

        # Return structure
        return {
            "action": action,
            "justification": justification.__dict__,
            "debug": {
                "candidate_count_initial": len(candidates),
                "candidate_count_scored": len(scored),
                "best_path_length": None if chosen is None else chosen.length()
            }
        }

    # ------------------------- Candidate Generation -------------------------

    def _generate_candidates(
        self,
        state: Dict[str, Any],
        start: Any,
        goal: Any
    ) -> Tuple[List[CandidatePath], int, bool]:
        """
        Bounded A* producing up to K candidate prefixes (length prefix_h).
        Returns (candidates, rejected_count, heuristic_augmented_flag)
        """
        use_neural = self.cfg.use_neural and self.neural is not None
        augmented = False
        base_drive_errors = self.drive_error_fn(state)

        # Potential neural augmentation
        if use_neural:
            pred = self.neural.predict_state(state)
            if pred.confidence >= self.cfg.confidence_min and pred.risk_score < self.cfg.risk_cap_for_heuristic:
                augmented = True
                neural_add = (
                    0.3 * pred.drive_cost +
                    0.5 * pred.risk_score +
                    0.2 * max(0.0, pred.energy_delta)
                )
            else:
                neural_add = 0.0
        else:
            neural_add = 0.0

        def heuristic(node: Any) -> float:
            # Simple Manhattan for (x,y) style tuples, else 0
            if isinstance(node, tuple) and isinstance(goal, tuple) and len(node) == 2 and len(goal) == 2:
                h = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
            else:
                h = 0.0
            if augmented:
                h += neural_add
            # Add a crude drive penalty heuristic: sum of abs errors
            h += 0.1 * sum(abs(e) for e in base_drive_errors.values())
            return h

        open_heap: List[Tuple[float, int, CandidatePath]] = []
        start_path = CandidatePath(
            nodes=[start],
            actions=[],
            heuristic=heuristic(start),
            g_cost=0.0
        )
        counter = 0
        heapq.heappush(open_heap, (start_path.heuristic, counter, start_path))
        counter += 1

        candidates: List[CandidatePath] = []
        visited_cost: Dict[Any, float] = {start: 0.0}
        expansions = 0
        rejected = 0

        while open_heap and len(candidates) < self.cfg.candidate_k and expansions < self.cfg.max_expansions:
            _, _, path = heapq.heappop(open_heap)
            current = path.nodes[-1]

            # Early pruning checks
            if self.is_principle_violation(current):
                rejected += 1
                continue
            if self.is_promise_violation(current):
                rejected += 1
                continue

            # If we reached goal or max prefix length satisfied
            if current == goal or path.length() == self.cfg.prefix_h:
                # Set violation flags for penalty calculation
                path.promise_violation = any(self.is_promise_violation(node) for node in path.nodes)
                path.principle_violation = any(self.is_principle_violation(node) for node in path.nodes)
                candidates.append(path)
                continue

            # Expand neighbors
            for nbr, action in self.get_neighbors(current):
                if nbr in path.nodes:
                    # avoid cycles
                    continue
                g_new = path.g_cost + 1.0  # uniform step cost here
                if nbr in visited_cost and g_new >= visited_cost[nbr]:
                    continue
                visited_cost[nbr] = g_new
                new_path = CandidatePath(
                    nodes=path.nodes + [nbr],
                    actions=path.actions + [action],
                    heuristic=heuristic(nbr),
                    g_cost=g_new
                )
                # Set violation flags for penalty calculation
                new_path.promise_violation = any(self.is_promise_violation(node) for node in new_path.nodes)
                new_path.principle_violation = any(self.is_principle_violation(node) for node in new_path.nodes)
                heapq.heappush(open_heap, (new_path.heuristic + g_new, counter, new_path))
                counter += 1
            expansions += 1

        return candidates, rejected, augmented

    # --------------------------- Candidate Scoring --------------------------

    def _score_candidates(
        self,
        state: Dict[str, Any],
        candidates: List[CandidatePath]
    ) -> List[Tuple[float, CandidatePath]]:
        scored: List[Tuple[float, CandidatePath]] = []

        for cand in candidates:
            # Drive-aware projection:
            # Start with current aggregate drive cost
            base_drive_cost_current = self._heuristic_drive_cost(state)
            # Simple projection of energy drift: assume each step consumes 0.01 energy units when energy below setpoint
            drive_errors_now = self.drive_error_fn(state)
            energy_err = drive_errors_now.get("energy", 0.0)  # (current - setpoint)
            projected_energy_err = energy_err + 0.01 * cand.total_steps()
            # Only penalize further deviation if squared error worsens
            if abs(projected_energy_err) > abs(energy_err):
                proj_energy_component = projected_energy_err**2 - energy_err**2
            else:
                proj_energy_component = 0.0
            if self.imagination:
                # Use imagination rollouts for enhanced prediction
                rollout_result = self.imagination.rollout_candidate(
                    initial_state=state,
                    candidate_path=cand,
                    drive_system=self.drive_system,
                    get_neighbors_fn=self.get_neighbors
                )
                
                if rollout_result["success"]:
                    # Use rollout predictions
                    total_drive_cost = sum(abs(delta) for delta in rollout_result["drive_deltas"].values())
                    cand.predicted_drive_cost = base_drive_cost_current + total_drive_cost
                    cand.predicted_risk_cost = rollout_result["risk_score"]
                    cand.predicted_energy_delta = -rollout_result["total_energy_consumed"]
                else:
                    # Fallback to heuristic if rollout failed
                    cand.predicted_drive_cost = base_drive_cost_current + proj_energy_component
                    cand.predicted_risk_cost = 0.0
                    cand.predicted_energy_delta = -0.01 * cand.total_steps()
                    
            elif self.neural and self.cfg.use_neural:
                # Neural path prediction supplements projected base cost
                pred = self.neural.predict_path(state, cand)
                cand.predicted_drive_cost = base_drive_cost_current + proj_energy_component + pred.drive_cost
                cand.predicted_risk_cost = pred.risk_score
                cand.predicted_energy_delta = pred.energy_delta
            else:
                cand.predicted_drive_cost = base_drive_cost_current + proj_energy_component
                cand.predicted_risk_cost = 0.0
                # Nominal energy usage estimate (negative means consumption)
                cand.predicted_energy_delta = -0.01 * cand.total_steps()

            # Hard filters should already be handled; recheck for safety
            if cand.promise_violation or cand.principle_violation:
                continue

            # MPC-style aggregate cost
            J = (
                self.weights.alpha * cand.total_steps() +
                self.weights.beta * cand.predicted_drive_cost +
                self.weights.gamma * cand.predicted_risk_cost +
                self.weights.delta * max(0.0, cand.predicted_energy_delta)
            )

            # Calculate promise penalties - both hard violations and proximity risks
            promise_pen = 0.0
            if self.promise_book:
                # Hard violations should be filtered out by early pruning,
                # but we can add proximity penalties for paths that get close to violations
                for promise in self.promise_book.active_promises():
                    penalty_value = self.promise_book.penalty_value(promise['id'])
                    if penalty_value > 0:
                        for node in cand.nodes:
                            if self.promise_book.node_violates(node):
                                # This shouldn't happen due to early pruning, but add penalty if it does
                                promise_pen += penalty_value
                                break
            
            principle_pen = self.weights.mu if cand.principle_violation else 0.0
            J += promise_pen + principle_pen

            scored.append((J, cand))

        scored.sort(key=lambda x: x[0])
        return scored

    # ---------------------------- Selection Logic ---------------------------

    def _select_best(
        self,
        scored: List[Tuple[float, CandidatePath]]
    ) -> Tuple[Optional[CandidatePath], Optional[str]]:
        if not scored:
            # fallback = none
            return None, "no_viable_candidates"
        # Best path = minimal cost
        return scored[0][1], None

    # --------------------------- Justification ------------------------------

    def _build_justification(
        self,
        state: Dict[str, Any],
        chosen: Optional[CandidatePath],
        rejected: int,
        augmented: bool,
        fallback_reason: Optional[str]
    ) -> Justification:
        drive_errors = self.drive_error_fn(state)
        dominant_drive = None
        max_abs = 0.0
        for k, v in drive_errors.items():
            av = abs(v)
            if av > max_abs:
                max_abs = av
                dominant_drive = k

        # Determine sacrificed metric placeholder
        sacrificed_metric = None
        if chosen and chosen.total_steps() > 0 and max_abs > 0.5:
            sacrificed_metric = "path_length_over_optimal"

        cost_breakdown = {}
        if chosen:
            cost_breakdown = {
                "steps": chosen.total_steps(),
                "drive_cost": chosen.predicted_drive_cost,
                "risk": chosen.predicted_risk_cost,
                "energy": chosen.predicted_energy_delta,
                "promise_penalties": self.weights.lambd if chosen.promise_violation else 0.0,
                "principle_penalties": self.weights.mu if chosen.principle_violation else 0.0
            }

        model_confidence = None
        model_used = False
        if self.neural and self.cfg.use_neural and augmented:
            # We do not store per-path confidence separately here; stub value
            model_confidence = 0.75  # placeholder
            model_used = True

        promise_status = "no_conflict"
        if chosen and chosen.promise_violation:
            promise_status = "violation_discarded"

        return Justification(
            chosen_goal="goal_reached" if (chosen and chosen.nodes and chosen.nodes[-1] == self.goal_fn(state)) else "progress",
            principles_checked=["do_not_harm", "keep_promises", "conserve_energy"],  # placeholder top 3
            rejected_candidates_count=rejected,
            dominant_drive=dominant_drive,
            sacrificed_metric=sacrificed_metric,
            promise_status=promise_status,
            model_confidence=model_confidence,
            model_used=model_used,
            heuristic_augmented=augmented,
            fallback_reason=fallback_reason,
            cost_breakdown=cost_breakdown
        )

    # ---------------------------- Utility Methods ---------------------------

    def _heuristic_drive_cost(self, state: Dict[str, Any]) -> float:
        errors = self.drive_error_fn(state)
        # Weighted sum of squared errors (weights unknown here; simple 1.0)
        return sum(e * e for e in errors.values())

    def _noop_justified(self, reason: str) -> Dict[str, Any]:
        justification = Justification(
            chosen_goal=None,
            principles_checked=["do_not_harm", "keep_promises", "conserve_energy"],
            rejected_candidates_count=0,
            dominant_drive=None,
            sacrificed_metric=None,
            promise_status="no_goal",
            model_confidence=None,
            model_used=False,
            heuristic_augmented=False,
            fallback_reason=reason,
            cost_breakdown={}
        )
        return {
            "action": {"type": "noop"},
            "justification": justification.__dict__,
            "debug": {}
        }


# ---------------------------------------------------------------------------
# Simple Reference Implementations / Defaults
# ---------------------------------------------------------------------------

def default_get_neighbors(node: Any) -> List[Tuple[Any, Any]]:
    """Assumes 2D grid node (x,y). Returns cardinal neighbors with actions."""
    if not (isinstance(node, tuple) and len(node) == 2):
        return []
    x, y = node
    moves = [
        ((x + 1, y), {"type": "move", "dx": 1, "dy": 0}),
        ((x - 1, y), {"type": "move", "dx": -1, "dy": 0}),
        ((x, y + 1), {"type": "move", "dx": 0, "dy": 1}),
        ((x, y - 1), {"type": "move", "dx": 0, "dy": -1}),
    ]
    return moves


def always_safe(_node: Any) -> bool:
    return False


def dummy_drive_errors(_state: Dict[str, Any]) -> Dict[str, float]:
    return {"energy": 0.0, "temperature": 0.0, "social_proximity": 0.0}


def simple_goal_fn(_state: Dict[str, Any]) -> Optional[Any]:
    # Placeholder: expects state["goal"] if present
    return _state.get("goal")


def extract_pos(state: Dict[str, Any]) -> Any:
    return state.get("agent_pos", (0, 0))


def make_default_planner() -> Planner:
    return Planner(
        config=PlannerConfig(),
        cost_weights=CostWeights(),
        get_neighbors=default_get_neighbors,
        is_principle_violation=always_safe,
        is_promise_violation=always_safe,
        drive_error_fn=dummy_drive_errors,
        goal_fn=simple_goal_fn,
        state_position_extractor=extract_pos,
        neural_module=None
    )


# If run standalone (manual quick test)
if __name__ == "__main__":
    planner = make_default_planner()
    test_state = {"agent_pos": (0, 0), "goal": (3, 0)}
    result = planner.plan(test_state)
    print("Planner test output:", result)
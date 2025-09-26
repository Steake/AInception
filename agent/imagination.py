"""
Imagination Module: Counterfactual Rollouts for MPC-style Planning

Responsibilities:
- Simulate action sequences using a forward model
- Predict drive state changes, energy consumption, and risk accumulation
- Support multiple horizons (typically 1 and 3 steps)
- Cache predictions for surprise detection
- Provide cost estimates for planner candidate evaluation

Forward Model Assumptions (Simplified for MVP):
- Grid-based movement with energy consumption per step
- Drive dynamics: energy decreases, temperature/social drift toward environmental values
- Risk accumulation from dangerous tiles or forbidden zones
- Deterministic simulation for consistency

Integration with Planner:
- Called from _score_candidates to enhance cost prediction
- Replaces simple heuristics with multi-step lookahead
- Feeds into surprise detection by comparing predicted vs actual outcomes

Rollout Structure:
{
  "horizon": int,
  "action_sequence": List[Any],
  "predicted_states": List[Dict[str, Any]],
  "drive_deltas": Dict[str, float],
  "total_energy_consumed": float,
  "risk_score": float,
  "final_drive_errors": Dict[str, float]
}
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import copy


class Imagination:
    def __init__(self, horizons: Tuple[int, ...] = (1, 3)):
        """
        Initialize imagination module with specified rollout horizons.
        
        Args:
            horizons: Tuple of horizon lengths to simulate (e.g., (1, 3))
        """
        self.horizons = horizons
        self._prediction_cache: Dict[str, Dict[str, Any]] = {}
    
    def rollout(
        self,
        initial_state: Dict[str, Any],
        action_sequence: List[Any],
        drive_system: Any,
        get_neighbors_fn: Any,
        horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Simulate an action sequence and predict outcomes.
        
        Args:
            initial_state: Current world state with agent position, drives, etc.
            action_sequence: Sequence of actions to simulate
            drive_system: DriveSystem instance for drive dynamics
            get_neighbors_fn: Function to get valid moves from a position
            horizon: Number of steps to simulate (truncated if action_sequence is shorter)
            
        Returns:
            Dictionary with predicted outcomes including drive changes and risk
        """
        # Limit simulation to specified horizon
        actions = action_sequence[:horizon]
        if not actions:
            return self._empty_rollout(initial_state, horizon)
        
        # Initialize simulation state
        current_state = copy.deepcopy(initial_state)
        predicted_states = [copy.deepcopy(current_state)]
        total_energy_consumed = 0.0
        risk_score = 0.0
        
        # Simulate each action
        for step, action in enumerate(actions):
            # Apply action to get new position
            new_state = self._apply_action(current_state, action, get_neighbors_fn)
            if new_state is None:
                # Invalid action, stop simulation
                break
                
            # Update drive dynamics
            drive_effects = self._simulate_drive_dynamics(new_state, drive_system)
            new_state.update(drive_effects)
            
            # Calculate energy consumption
            energy_cost = self._calculate_energy_cost(current_state, new_state, action)
            total_energy_consumed += energy_cost
            new_state["energy"] = current_state.get("energy", 1.0) - energy_cost
            
            # Accumulate risk from environment
            step_risk = self._calculate_step_risk(new_state)
            risk_score += step_risk
            
            predicted_states.append(copy.deepcopy(new_state))
            current_state = new_state
        
        # Calculate drive deltas
        initial_drives = self._extract_drive_values(initial_state)
        final_drives = self._extract_drive_values(current_state)
        drive_deltas = {
            drive: final_drives.get(drive, 0.0) - initial_drives.get(drive, 0.0)
            for drive in initial_drives.keys()
        }
        
        # Calculate final drive errors
        final_drive_errors = drive_system.drive_errors() if drive_system else {}
        
        return {
            "horizon": len(actions),
            "action_sequence": actions,
            "predicted_states": predicted_states,
            "drive_deltas": drive_deltas,
            "total_energy_consumed": total_energy_consumed,
            "risk_score": risk_score,
            "final_drive_errors": final_drive_errors,
            "success": len(predicted_states) > 1  # At least one step simulated
        }
    
    def rollout_candidate(
        self,
        initial_state: Dict[str, Any],
        candidate_path: Any,
        drive_system: Any,
        get_neighbors_fn: Any
    ) -> Dict[str, Any]:
        """
        Convenience method to rollout a candidate path.
        Uses the longest configured horizon or the path length, whichever is shorter.
        """
        max_horizon = max(self.horizons) if self.horizons else 3
        horizon = min(max_horizon, len(candidate_path.actions))
        
        return self.rollout(
            initial_state=initial_state,
            action_sequence=candidate_path.actions,
            drive_system=drive_system,
            get_neighbors_fn=get_neighbors_fn,
            horizon=horizon
        )
    
    def cache_prediction(self, state_key: str, action: Any, prediction: Dict[str, Any]):
        """Cache a prediction for later surprise detection."""
        self._prediction_cache[f"{state_key}:{action}"] = prediction
    
    def get_cached_prediction(self, state_key: str, action: Any) -> Optional[Dict[str, Any]]:
        """Retrieve a cached prediction."""
        return self._prediction_cache.get(f"{state_key}:{action}")
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self._prediction_cache.clear()
    
    # ----------------------------- Private Methods -----------------------------
    
    def _empty_rollout(self, initial_state: Dict[str, Any], horizon: int) -> Dict[str, Any]:
        """Return empty rollout when no actions provided."""
        return {
            "horizon": 0,
            "action_sequence": [],
            "predicted_states": [copy.deepcopy(initial_state)],
            "drive_deltas": {},
            "total_energy_consumed": 0.0,
            "risk_score": 0.0,
            "final_drive_errors": {},
            "success": True
        }
    
    def _apply_action(
        self,
        state: Dict[str, Any],
        action: Any,
        get_neighbors_fn: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Apply an action to the current state and return new state.
        Returns None if action is invalid.
        """
        current_pos = state.get("agent_pos", (0, 0))
        
        # Handle different action formats
        if isinstance(action, dict):
            if action.get("type") == "move":
                new_pos = action.get("position")
                if new_pos is None:
                    return None
            elif action.get("type") == "noop":
                new_pos = current_pos
            else:
                return None
        else:
            # Assume action is a position tuple
            new_pos = action
        
        # Validate move is legal using get_neighbors_fn
        if new_pos != current_pos:
            neighbors = get_neighbors_fn(current_pos)
            valid_positions = [neighbor for neighbor, _ in neighbors]
            if new_pos not in valid_positions:
                return None
        
        # Create new state with updated position
        new_state = copy.deepcopy(state)
        new_state["agent_pos"] = new_pos
        return new_state
    
    def _simulate_drive_dynamics(self, state: Dict[str, Any], drive_system: Any) -> Dict[str, Any]:
        """
        Simulate how drives change based on the new state.
        Simplified model: drives drift toward environmental values.
        """
        effects = {}
        
        # Temperature drift (simplified)
        env_temp = state.get("temperature", 0.5)
        current_temp = state.get("agent_temperature", 0.5)
        temp_drift = (env_temp - current_temp) * 0.1  # 10% drift per step
        effects["agent_temperature"] = current_temp + temp_drift
        
        # Social proximity (simplified - decays without social contact)
        current_social = state.get("social_proximity", 0.5)
        social_decay = 0.05  # 5% decay per step
        effects["social_proximity"] = max(0.0, current_social - social_decay)
        
        return effects
    
    def _calculate_energy_cost(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        action: Any
    ) -> float:
        """Calculate energy cost of an action."""
        old_pos = old_state.get("agent_pos", (0, 0))
        new_pos = new_state.get("agent_pos", (0, 0))
        
        # Base energy cost for movement
        if old_pos != new_pos:
            return 0.01  # Standard movement cost
        else:
            return 0.001  # Idle cost
    
    def _calculate_step_risk(self, state: Dict[str, Any]) -> float:
        """Calculate risk accumulated in current state."""
        risk = 0.0
        current_pos = state.get("agent_pos", (0, 0))
        
        # Risk from danger tiles
        danger_tiles = state.get("danger_tiles", set())
        if current_pos in danger_tiles:
            risk += 0.5
        
        # Risk from forbidden tiles (should be caught by constraints)
        forbidden_tiles = state.get("forbidden_tiles", set())
        if current_pos in forbidden_tiles:
            risk += 1.0
        
        # Risk from extreme drive states
        energy = state.get("energy", 1.0)
        if energy < 0.1:
            risk += 0.3  # Low energy risk
        
        return risk
    
    def _extract_drive_values(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Extract drive values from state."""
        return {
            "energy": state.get("energy", 1.0),
            "temperature": state.get("agent_temperature", 0.5),
            "social_proximity": state.get("social_proximity", 0.5)
        }


# Example usage and testing
if __name__ == "__main__":
    # Simple test
    imagination = Imagination(horizons=(1, 3))
    
    initial_state = {
        "agent_pos": (0, 0),
        "energy": 1.0,
        "agent_temperature": 0.5,
        "social_proximity": 0.8,
        "danger_tiles": {(2, 2)},
        "forbidden_tiles": {(3, 3)}
    }
    
    def simple_neighbors(pos):
        x, y = pos
        return [((x+1, y), {"type": "move", "position": (x+1, y)}),
                ((x, y+1), {"type": "move", "position": (x, y+1)})]
    
    actions = [{"type": "move", "position": (1, 0)},
               {"type": "move", "position": (2, 0)},
               {"type": "move", "position": (2, 1)}]
    
    result = imagination.rollout(
        initial_state=initial_state,
        action_sequence=actions,
        drive_system=None,
        get_neighbors_fn=simple_neighbors,
        horizon=3
    )
    
    print("Rollout result:")
    print(f"Energy consumed: {result['total_energy_consumed']}")
    print(f"Risk score: {result['risk_score']}")
    print(f"Drive deltas: {result['drive_deltas']}")
    print(f"Success: {result['success']}")

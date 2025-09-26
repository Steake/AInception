"""
Unit tests for Imagination - rollout prediction system.
"""
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agent.imagination import Imagination
from agent.drives import DriveSystem


class TestImagination(unittest.TestCase):
    
    def setUp(self):
        self.imagination = Imagination(horizons=(1, 3))
        
        # Create default drive specs for testing
        drive_specs = {
            "energy": {
                "setpoint": 1.0,
                "weight": 1.0,
                "initial": 1.0,
                "min": 0.0,
                "max": 1.0,
                "decay_rate": 0.01
            },
            "temperature": {
                "setpoint": 0.5,
                "weight": 1.0,
                "initial": 0.5,
                "min": 0.0,
                "max": 1.0,
                "decay_rate": 0.0
            },
            "social_proximity": {
                "setpoint": 0.0,
                "weight": 1.0,
                "initial": 0.0,
                "min": 0.0,
                "max": 1.0,
                "decay_rate": 0.05
            }
        }
        self.drives = DriveSystem(drive_specs)
    
    def test_single_step_rollout(self):
        """Test single step action rollout."""
        state = {
            "agent_pos": (2, 2),
            "energy": 0.8,
            "agent_temperature": 0.6,
            "social_proximity": 0.5
        }
        
        def simple_neighbors(pos):
            x, y = pos
            return [((x+1, y), {"type": "move", "position": (x+1, y)})]
        
        action_sequence = [{"type": "move", "position": (3, 2)}]
        
        result = self.imagination.rollout(
            initial_state=state,
            action_sequence=action_sequence,
            drive_system=self.drives,
            get_neighbors_fn=simple_neighbors,
            horizon=1
        )
        
        # Check result structure
        self.assertIn('drive_deltas', result)
        self.assertIn('total_energy_consumed', result)
        self.assertIn('risk_score', result)
        self.assertIn('predicted_states', result)
        
        # Check drive deltas structure
        self.assertIn('energy', result['drive_deltas'])
        self.assertIn('temperature', result['drive_deltas'])
        self.assertIn('social_proximity', result['drive_deltas'])
    
    def test_multi_step_rollout(self):
        """Test multi-step action sequence rollout."""
        state = {
            "agent_pos": (1, 1),
            "energy": 1.0,
            "agent_temperature": 0.5,
            "social_proximity": 0.5
        }
        
        def simple_neighbors(pos):
            x, y = pos
            return [
                ((x+1, y), {"type": "move", "position": (x+1, y)}),
                ((x, y+1), {"type": "move", "position": (x, y+1)})
            ]
        
        action_sequence = [
            {"type": "move", "position": (2, 1)},
            {"type": "move", "position": (2, 2)},
            {"type": "move", "position": (3, 2)}
        ]
        
        result = self.imagination.rollout(
            initial_state=state,
            action_sequence=action_sequence,
            drive_system=self.drives,
            get_neighbors_fn=simple_neighbors,
            horizon=3
        )
        
        # Multi-step should have larger energy consumption
        self.assertGreater(result['total_energy_consumed'], 0.02)  # More consumption than single step
        
        # Final position should reflect all moves if successful
        if result['success'] and len(result['predicted_states']) > 3:
            expected_final_pos = (3, 2)  # Final destination
            final_state = result['predicted_states'][-1]
            self.assertEqual(final_state['agent_pos'], expected_final_pos)
    
    def test_drive_projection(self):
        """Test drive state projection accuracy."""
        state = {
            "agent_pos": (0, 0),
            "energy": 0.5,  # Below target
            "agent_temperature": 0.8,  # Above target
            "social_proximity": 0.2
        }
        
        def simple_neighbors(pos):
            x, y = pos
            return [((x+1, y), {"type": "move", "position": (x+1, y)})]
        
        action_sequence = [{"type": "move", "position": (1, 0)}]
        
        result = self.imagination.rollout(
            initial_state=state,
            action_sequence=action_sequence,
            drive_system=self.drives,
            get_neighbors_fn=simple_neighbors,
            horizon=1
        )
        
        # Energy should decrease (movement cost)
        self.assertGreater(result['total_energy_consumed'], 0)  # Positive means consumption
        
        # Temperature delta depends on environment model
        self.assertIn('temperature', result['drive_deltas'])
    
    def test_risk_assessment(self):
        """Test risk score calculation."""
        # Safe state
        safe_state = {
            "agent_pos": (1, 1),
            "energy": 0.9,
            "agent_temperature": 0.5,
            "social_proximity": 0.5,
            "danger_tiles": {(5, 5)}  # Far from danger
        }
        
        def simple_neighbors(pos):
            x, y = pos
            return [((x+1, y), {"type": "move", "position": (x+1, y)})]
        
        safe_result = self.imagination.rollout(
            initial_state=safe_state,
            action_sequence=[{"type": "move", "position": (2, 1)}],
            drive_system=self.drives,
            get_neighbors_fn=simple_neighbors,
            horizon=1
        )
        
        # Risky state
        risky_state = {
            "agent_pos": (4, 5),
            "energy": 0.2,  # Low energy
            "agent_temperature": 0.9,  # High temperature
            "social_proximity": 0.1,
            "danger_tiles": {(5, 5)}  # Near danger
        }
        
        risky_result = self.imagination.rollout(
            initial_state=risky_state,
            action_sequence=[{"type": "move", "position": (5, 5)}],  # Move to danger
            drive_system=self.drives,
            get_neighbors_fn=simple_neighbors,
            horizon=1
        )
        
        # Risky state should have higher risk score
        self.assertGreaterEqual(risky_result['risk_score'], safe_result['risk_score'])
    
    def test_different_horizons(self):
        """Test rollout with different horizon lengths."""
        state = {
            "agent_pos": (0, 0),
            "energy": 1.0,
            "agent_temperature": 0.5,
            "social_proximity": 0.5
        }
        
        def simple_neighbors(pos):
            x, y = pos
            return [((x+1, y), {"type": "move", "position": (x+1, y)})]
        
        short_sequence = [{"type": "move", "position": (1, 0)}]
        long_sequence = [
            {"type": "move", "position": (1, 0)},
            {"type": "move", "position": (2, 0)},
            {"type": "move", "position": (3, 0)},
            {"type": "move", "position": (4, 0)}
        ]
        
        short_result = self.imagination.rollout(
            initial_state=state,
            action_sequence=short_sequence,
            drive_system=self.drives,
            get_neighbors_fn=simple_neighbors,
            horizon=1
        )
        
        long_result = self.imagination.rollout(
            initial_state=state,
            action_sequence=long_sequence,
            drive_system=self.drives,
            get_neighbors_fn=simple_neighbors,
            horizon=4
        )
        
        # Longer sequence should have more energy consumption
        self.assertGreater(long_result['total_energy_consumed'], short_result['total_energy_consumed'])


if __name__ == '__main__':
    unittest.main()
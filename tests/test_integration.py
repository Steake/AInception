"""
Integration tests for component interactions.
"""
import unittest
import sys
import tempfile
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agent.core import Agent
from worlds.gridworld import GridWorld


class TestAgentIntegration(unittest.TestCase):
    
    def setUp(self):
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.agent = Agent(enable_journal_llm=False, db_path=self.temp_db.name)
        
        self.world = GridWorld(
            width=5, height=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            danger_tiles={(2, 2)},
            forbidden_tiles={(1, 1)}
        )
    
    def tearDown(self):
        # Clean up temporary database
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_agent_world_interaction(self):
        """Test basic agent-world interaction loop."""
        self.world.reset()
        observation = self.world.get_observation()
        
        # Agent should produce valid action
        result = self.agent.step(observation)
        
        self.assertIn('action', result)
        self.assertIn('justification', result)
        
        action = result['action']
        self.assertIn('type', action)
        
        # Action should be one of the valid types
        valid_actions = ['move', 'idle', 'noop']
        self.assertIn(action['type'], valid_actions)
    
    def test_promise_enforcement(self):
        """Test that agent respects promises during planning."""
        # Register promise to avoid forbidden tile
        promise_id = self.agent.register_promise(
            condition="avoid:(1,1)",
            behavior="Do not enter tile (1,1)",
            expiry=100,
            penalty="cost:50.0"
        )
        
        self.world.reset()
        observation = self.world.get_observation()
        
        # Run several steps
        violation_detected = False
        for step in range(10):
            result = self.agent.step(observation)
            action = result['action']
            
            # Apply action
            world_result = self.world.step(action)
            observation = world_result['observation']
            
            # Agent should never be on forbidden tile
            agent_pos = observation['agent_pos']
            if agent_pos == (1, 1):
                violation_detected = True
                break
                
            if world_result.get('done', False):
                break
        
        self.assertFalse(violation_detected, "Agent violated promise by entering forbidden tile")
    
    def test_drive_management(self):
        """Test that agent manages drives appropriately."""
        self.world.reset()
        observation = self.world.get_observation()
        
        initial_drives = self.agent.drives.drive_errors()
        
        # Run episode
        for step in range(20):
            result = self.agent.step(observation)
            
            # Check drives are being tracked
            current_drives = self.agent.drives.drive_errors()
            self.assertIsInstance(current_drives, dict)
            self.assertIn('energy', current_drives)
            
            # Apply action
            world_result = self.world.step(result['action'])
            observation = world_result['observation']
            
            if world_result.get('done'):
                break
    
    def test_constitutional_principles(self):
        """Test that constitutional principles influence decisions."""
        # Get baseline decision pattern
        self.world.reset()
        observation = self.world.get_observation()
        
        result = self.agent.step(observation)
        justification = result['justification']
        
        # Justification should reference principles being checked
        self.assertIn('principles_checked', justification)
        self.assertIsInstance(justification['principles_checked'], list)
        self.assertGreater(len(justification['principles_checked']), 0)
    
    def test_reflex_layer_activation(self):
        """Test that reflex layer activates appropriately."""
        # Set up low energy scenario
        self.agent.drives.set_drive('energy', 0.2)  # Very low energy
        
        self.world.reset()
        observation = self.world.get_observation()
        
        result = self.agent.step(observation)
        
        # Should either idle or choose energy-preserving action
        action = result['action']
        reflex_used = result['justification'].get('reflex_triggered', False)
        
        # In low energy state, reflex should often trigger or choose idle
        if self.agent.drives.drives['energy'].current < 0.2:
            self.assertTrue(reflex_used or action.get('type') == 'idle')


if __name__ == '__main__':
    unittest.main()
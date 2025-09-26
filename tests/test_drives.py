"""
Unit tests for DriveSystem - homeostatic drive management.
"""
import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agent.drives import DriveSystem


class TestDriveSystem(unittest.TestCase):
    
    def setUp(self):
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
    
    def test_drive_initialization(self):
        """Test drives initialize to target values."""
        self.assertAlmostEqual(self.drives.drives['energy'].current, 1.0, places=2)
        self.assertAlmostEqual(self.drives.drives['temperature'].current, 0.5, places=2)
        self.assertAlmostEqual(self.drives.drives['social_proximity'].current, 0.0, places=2)
    
    def test_energy_update(self):
        """Test energy updates with consumption."""
        initial_energy = self.drives.drives['energy'].current
        
        # Update via observation
        self.drives.ingest_observation({'energy': 0.8})
        
        self.assertLess(self.drives.drives['energy'].current, initial_energy)
        self.assertGreaterEqual(self.drives.drives['energy'].current, 0.0)  # Should not go negative
    
    def test_temperature_update(self):
        """Test temperature updates."""
        self.drives.ingest_observation({'temperature': 0.8})  # Set high temperature
        self.assertAlmostEqual(self.drives.drives['temperature'].current, 0.8, places=2)
        
        # Test clamping
        self.drives.ingest_observation({'temperature': 1.5})  # Above max
        self.assertLessEqual(self.drives.drives['temperature'].current, 1.0)
    
    def test_drive_errors(self):
        """Test drive error calculation."""
        # Set drives away from targets
        self.drives.ingest_observation({'energy': 0.7, 'temperature': 0.9})  # Low energy, high temperature
        
        errors = self.drives.drive_errors()
        
        self.assertIn('energy', errors)
        self.assertIn('temperature', errors)
        self.assertIn('social_proximity', errors)
        
        # Energy should show negative error (below target)
        self.assertLess(errors['energy'], 0)
        # Temperature should show positive error (above target)
        self.assertGreater(errors['temperature'], 0)
    
    def test_projection_utility(self):
        """Test drive projection utility calculation."""
        # Test energy projection
        current_energy = self.drives.drives['energy'].current
        projected_cost = self.drives.project_with_deltas({'energy': -0.2})
        current_cost = self.drives.total_cost()
        
        # Should increase cost since moving away from target
        self.assertGreater(projected_cost, current_cost)
        
        # Test closer to target should have less cost increase
        closer_cost = self.drives.project_with_deltas({'energy': -0.05})
        self.assertLess(closer_cost, projected_cost)


if __name__ == '__main__':
    unittest.main()
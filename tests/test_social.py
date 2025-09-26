"""
Unit tests for PromiseBook - social promise management.
"""
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agent.social import PromiseBook


class TestPromiseBook(unittest.TestCase):
    
    def setUp(self):
        self.promises = PromiseBook()
        self.test_logs = []
        
        # Set up logging callback
        def log_callback(record):
            self.test_logs.append(record)
        
        self.promises = PromiseBook(change_log_cb=log_callback)
    
    def test_promise_registration(self):
        """Test promise registration and structure."""
        promise_id = self.promises.register(
            condition="avoid:(5,5)",
            behavior="Do not enter tile (5,5)",
            expiry=100,
            penalty="cost:10.0"
        )
        
        self.assertIsInstance(promise_id, str)
        self.assertGreater(len(promise_id), 0)
        
        # Check promise structure
        promise = self.promises.get(promise_id)
        self.assertIsNotNone(promise)
        self.assertEqual(promise['condition'], "avoid:(5,5)")
        self.assertEqual(promise['behavior'], "Do not enter tile (5,5)")
        self.assertEqual(promise['status'], 'active')
    
    def test_avoid_condition_parsing(self):
        """Test avoid condition parsing and violation detection."""
        promise_id = self.promises.register(
            condition="avoid:(3,7)",
            behavior="Stay away from (3,7)",
            expiry=50,
            penalty="cost:5.0"
        )
        
        # Test violation detection
        self.assertTrue(self.promises.node_violates((3, 7)))
        self.assertFalse(self.promises.node_violates((3, 6)))
        self.assertFalse(self.promises.node_violates((2, 7)))
    
    def test_penalty_calculation(self):
        """Test penalty value extraction."""
        promise_id = self.promises.register(
            condition="avoid:(1,1)",
            behavior="Test promise",
            expiry=100,
            penalty="cost:15.5"
        )
        
        penalty = self.promises.penalty_value(promise_id)
        self.assertAlmostEqual(penalty, 15.5, places=1)
        
        # Test unknown penalty pattern
        promise_id2 = self.promises.register(
            condition="avoid:(2,2)",
            behavior="Test promise 2",
            expiry=100,
            penalty="unknown:something"
        )
        
        penalty2 = self.promises.penalty_value(promise_id2)
        self.assertEqual(penalty2, 0.0)  # Should default to 0
    
    def test_promise_lifecycle(self):
        """Test promise lifecycle management."""
        promise_id = self.promises.register(
            condition="avoid:(4,4)",
            behavior="Test lifecycle",
            expiry=10,
            penalty="cost:1.0"
        )
        
        # Initially active
        self.assertEqual(self.promises.get(promise_id)['status'], 'active')
        
        # Mark fulfilled
        self.promises.mark_fulfilled(promise_id, tick=5)
        self.assertEqual(self.promises.get(promise_id)['status'], 'fulfilled')
        self.assertEqual(self.promises.get(promise_id)['fulfill_tick'], 5)
        
        # Reset for breach test
        self.promises.get(promise_id)['status'] = 'active'
        self.promises.get(promise_id)['fulfill_tick'] = None
        
        # Mark breached
        self.promises.breach(promise_id, tick=8, reason="Entered forbidden tile")
        self.assertEqual(self.promises.get(promise_id)['status'], 'breached')
        self.assertEqual(self.promises.get(promise_id)['breach_tick'], 8)
    
    def test_expiry_handling(self):
        """Test automatic promise expiry."""
        promise_id = self.promises.register(
            condition="avoid:(0,0)",
            behavior="Test expiry",
            expiry=5,
            penalty="cost:1.0"
        )
        
        # Before expiry
        self.promises.update_tick(3)
        self.assertEqual(self.promises.get(promise_id)['status'], 'active')
        
        # After expiry
        self.promises.update_tick(10)
        self.assertEqual(self.promises.get(promise_id)['status'], 'expired')
    
    def test_breach_detection(self):
        """Test automatic breach detection."""
        promise_id = self.promises.register(
            condition="avoid:(6,6)",
            behavior="Test breach detection",
            expiry=100,
            penalty="cost:2.0"
        )
        
        # No breach initially
        self.assertEqual(self.promises.get(promise_id)['status'], 'active')
        
        # Trigger breach
        self.promises.mark_breach_if_needed((6, 6), tick=15)
        self.assertEqual(self.promises.get(promise_id)['status'], 'breached')
        self.assertEqual(self.promises.get(promise_id)['breach_tick'], 15)
    
    def test_serialization(self):
        """Test promise book serialization."""
        # Add some promises
        p1 = self.promises.register("avoid:(1,1)", "Test 1", 50, "cost:1.0")
        p2 = self.promises.register("avoid:(2,2)", "Test 2", 100, "cost:2.0")
        
        # Modify one promise
        self.promises.mark_fulfilled(p1, tick=25)
        
        # Serialize
        data = self.promises.to_dict()
        
        # Create new promise book and restore
        new_promises = PromiseBook()
        new_promises.from_dict(data)
        
        # Verify restoration
        self.assertEqual(len(new_promises.active_promises()), 1)  # Only p2 should be active
        self.assertEqual(new_promises.get(p1)['status'], 'fulfilled')
        self.assertEqual(new_promises.get(p2)['status'], 'active')


if __name__ == '__main__':
    unittest.main()
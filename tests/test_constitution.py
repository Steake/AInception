"""
Unit tests for Constitution - principle ranking and proof validation.
"""
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agent.constitution import Constitution


class TestConstitution(unittest.TestCase):
    
    def setUp(self):
        # Create default principles spec for testing
        principles_spec = [
            {"name": "safety_first", "description": "Prioritize safety above all", "rank": 1},
            {"name": "keep_promises", "description": "Honor social commitments", "rank": 2},
            {"name": "minimize_harm", "description": "Avoid causing harm", "rank": 3},
            {"name": "respect_autonomy", "description": "Respect others' autonomy", "rank": 4},
            {"name": "be_truthful", "description": "Be honest and transparent", "rank": 5}
        ]
        self.constitution = Constitution(principles_spec)
    
    def test_principle_loading(self):
        """Test principles load from config."""
        principles = self.constitution.top(5)
        self.assertGreater(len(principles), 0)
        
        # Check structure
        all_principles = self.constitution.principles_dict()
        for principle_name in principles:
            self.assertIn(principle_name, all_principles)
            principle = all_principles[principle_name]
            self.assertIn('name', principle)
            self.assertIn('description', principle)
            self.assertIn('rank', principle)
    
    def test_ranking_system(self):
        """Test principle ranking system."""
        ranking = self.constitution.ranking()
        
        # Should be sorted by rank
        ranks = [rank for rank, _ in ranking]
        self.assertEqual(ranks, sorted(ranks))
    
    def test_proof_validation(self):
        """Test proof-gated re-ranking."""
        # Valid proof structure with all required fields
        valid_proof = {
            "reason": "Extensive testing shows safety principle should be top priority",
            "tradeoffs": ["Higher energy cost for safety"],
            "timestamp": 1234567890,
            "evidence": "Multiple scenarios demonstrate critical importance",
            "affected_principles": ["safety_first", "keep_promises"]
        }
        
        # Get current order
        current_ranking = self.constitution.ranking()
        principle_names = [name for _, name in current_ranking]
        
        # Reorder to put safety_first at top
        new_order = ["safety_first"] + [name for name in principle_names if name != "safety_first"]
        
        # Should accept valid proof
        try:
            self.constitution.set_ranking(new_order, valid_proof)
            result = True
        except Exception:
            result = False
        self.assertTrue(result)
        
        # Invalid proof should be rejected (missing required fields)
        invalid_proof = {"evidence": "Because I said so"}
        try:
            self.constitution.set_ranking(new_order, invalid_proof)
            result = True
        except Exception:
            result = False
        self.assertFalse(result)
    
    def test_principle_evaluation(self):
        """Test principle evaluation on nodes."""
        # Test safe node
        safe_node = (1, 1)
        context = {"current_pos": safe_node}
        safe_eval = self.constitution.evaluate(context, top_k=3)
        self.assertIsInstance(safe_eval, dict)
        
        # Test dangerous node (if configured)
        danger_node = (3, 3)  # Common danger tile in tests
        context = {"current_pos": danger_node, "danger_tiles": {(3, 3)}}
        danger_eval = self.constitution.evaluate(context, top_k=3)
        self.assertIsInstance(danger_eval, dict)


if __name__ == '__main__':
    unittest.main()
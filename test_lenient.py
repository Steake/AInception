#!/usr/bin/env python3
"""
Alternative scenario test to verify agent completion with less aggressive energy conservation
"""
import tempfile
import os
from agent.core import Agent
from agent.policy.reflex import ReflexLayer
from worlds.gridworld import GridWorld

def test_with_lenient_energy():
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        agent = Agent(enable_journal_llm=False, db_path=temp_db.name)
        
        # Override the reflex layer with much more lenient energy conservation
        # Make energy threshold very aggressive (needs to be REALLY low to trigger)
        # and increase distance threshold so it rarely applies
        agent.reflex = ReflexLayer(
            energy_low_margin=0.35,  # Only trigger when 0.35 below setpoint (i.e. energy < 0.35)
            goal_distance_conserve_threshold=15,  # Only conserve if very far from goal
        )
        
        world = GridWorld(
            width=8, height=8,
            start_pos=(0, 0),
            goal_pos=(7, 7),
            danger_tiles={(3, 3), (5, 5)},
            forbidden_tiles=set()
        )
        
        world.reset()
        observation = world.get_observation()
        
        print("Testing with lenient energy conservation...")
        print(f"Initial state: agent={observation['agent_pos']}, item={observation.get('item_pos')}, goal={observation.get('goal')}")
        print(f"Initial energy: {observation.get('energy')}")
        print()
        
        max_steps = 100
        for step in range(max_steps):
            result = agent.step(observation)
            action = result['action']
            
            if step < 10 or step % 10 == 0:  # Print first 10 and every 10th step
                print(f"Step {step + 1}: {action} | Energy: {observation.get('energy'):.3f} | Agent: {observation['agent_pos']} | Carrying: {observation.get('carrying', False)}")
            
            # Apply action
            world_result = world.step(action)
            observation = world_result['observation']
            
            if world.check_goal_reached(observation):
                print(f"\n🎯 SUCCESS! Goal reached in {step + 1} steps")
                print(f"Final energy: {observation.get('energy'):.3f}")
                return True
        
        print(f"\n❌ FAILED: Did not reach goal in {max_steps} steps")
        print(f"Final energy: {observation.get('energy'):.3f}")
        return False
        
    finally:
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)

if __name__ == "__main__":
    success = test_with_lenient_energy()
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")
#!/usr/bin/env python3
"""
Debug script to understand scenario behavior
"""
import tempfile
import os
from agent.core import Agent
from worlds.gridworld import GridWorld

def debug_scenario():
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        agent = Agent(enable_journal_llm=False, db_path=temp_db.name)
        
        world = GridWorld(
            width=8, height=8,
            start_pos=(0, 0),
            goal_pos=(7, 7),
            danger_tiles={(3, 3), (5, 5)},
            forbidden_tiles=set()
        )
        
        world.reset()
        observation = world.get_observation()
        
        print("Initial state:")
        print(f"Agent pos: {observation['agent_pos']}")
        print(f"Item pos: {observation.get('item_pos', 'None')}")
        print(f"Goal pos: {observation.get('goal', 'None')}")
        print(f"Target pos (from world): {world.target_pos}")
        print(f"Carrying: {observation.get('carrying', False)}")
        print(f"Goal reached: {world.check_goal_reached(observation)}")
        print(f"Energy: {observation.get('energy', 'None')}")
        print()
        
        max_steps = 50
        for step in range(max_steps):
            print(f"=== Step {step + 1} ===")
            result = agent.step(observation)
            action = result['action']
            print(f"Action: {action}")
            
            # Apply action
            world_result = world.step(action)
            observation = world_result['observation']
            reward = world_result['reward']
            
            print(f"Agent pos: {observation['agent_pos']}")
            print(f"Item pos: {observation.get('item_pos', 'None')}")
            print(f"Goal pos: {observation.get('goal', 'None')}")
            print(f"Carrying: {observation.get('carrying', False)}")
            print(f"Delivered: {observation.get('delivered', False)}")
            print(f"Energy: {observation.get('energy', 'None')}")
            print(f"Reward: {reward}")
            print(f"Goal reached: {world.check_goal_reached(observation)}")
            print()
            
            if world.check_goal_reached(observation):
                print("🎯 GOAL REACHED!")
                break
                
            if step >= 20:  # Stop early for debug
                print("🛑 Stopping early for debug")
                break
        
        print(f"\nFinal goal reached: {world.check_goal_reached(observation)}")
        
    finally:
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)

if __name__ == "__main__":
    debug_scenario()
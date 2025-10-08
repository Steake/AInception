#!/usr/bin/env python3
"""
AInception Agent Comprehensive Demo

This script provides interactive demonstrations of the AInception agent's
capabilities, including:
- Navigation with homeostatic drives
- Constitutional principle enforcement
- Promise keeping and social contracts
- Adaptive behavior under perturbations
- Multi-constraint optimization

Usage:
    python demo_e2e.py --scenario full
    python demo_e2e.py --scenario promise
    python demo_e2e.py --scenario crisis
    python demo_e2e.py --scenario adaptive
    python demo_e2e.py --all
"""

import argparse
import sys
import tempfile
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agent.core import Agent
from worlds.gridworld import GridWorld


class DemoRunner:
    """Runner for comprehensive agent demonstrations."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = []
    
    def print_header(self, title: str):
        """Print formatted header."""
        if self.verbose:
            print("\n" + "="*80)
            print(f"DEMO: {title}")
            print("="*80 + "\n")
    
    def print_step(self, message: str):
        """Print step message."""
        if self.verbose:
            print(f"  {message}")
    
    def print_result(self, result: Dict[str, Any]):
        """Print formatted result."""
        if self.verbose:
            print("\n" + "-"*80)
            print("RESULTS:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            print("-"*80 + "\n")
    
    def demo_full_lifecycle(self):
        """
        Demonstration 1: Complete Agent Lifecycle
        
        Shows agent initialization, navigation, drive management,
        and goal achievement in a complex environment.
        """
        self.print_header("Complete Agent Lifecycle")
        
        # Setup
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        
        try:
            agent = Agent(enable_journal_llm=False, db_path=temp_db.name)
            self.print_step("✓ Agent initialized")
            
            world = GridWorld(
                width=12, height=12,
                start_pos=(0, 0),
                goal_pos=(11, 11),
                danger_tiles={(4, 4), (6, 6), (8, 8)},
                forbidden_tiles=set()
            )
            world.reset()
            self.print_step(f"✓ World created: 12x12 grid")
            self.print_step(f"  Start: {world._start_pos}, Goal: {world.target_pos}")
            self.print_step(f"  Danger zones: {len(world._danger_tiles)} tiles")
            
            # Run simulation
            observation = world.get_observation()
            initial_energy = observation['energy']
            path = [observation['agent_pos']]
            
            self.print_step("\nSimulation starting...")
            
            for step in range(200):
                result = agent.step(observation)
                world_result = world.step(result['action'])
                observation = world_result['observation']
                path.append(observation['agent_pos'])
                
                if step % 25 == 0:
                    self.print_step(f"Step {step:3d}: Pos {observation['agent_pos']}, "
                                   f"Energy {observation['energy']:.2f}")
                
                if world.check_goal_reached(observation):
                    self.print_step(f"\n✓ Goal reached at step {step}!")
                    break
            
            # Results
            result = {
                'scenario': 'full_lifecycle',
                'steps_taken': step + 1,
                'initial_energy': initial_energy,
                'final_energy': observation['energy'],
                'energy_consumed': initial_energy - observation['energy'],
                'goal_reached': world.check_goal_reached(observation),
                'path_length': len(path),
                'danger_zones_avoided': len(world._danger_tiles) - sum(1 for p in path if p in world._danger_tiles)
            }
            
            self.print_result(result)
            self.results.append(result)
            
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
    
    def demo_promise_keeping(self):
        """
        Demonstration 2: Promise Keeping Under Temptation
        
        Shows how the agent maintains promises even when breaking
        them would be more efficient.
        """
        self.print_header("Promise Keeping Under Temptation")
        
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        
        try:
            agent = Agent(enable_journal_llm=False, db_path=temp_db.name)
            
            # Register promises
            forbidden_positions = [(5, 5), (6, 6)]
            for pos in forbidden_positions:
                agent.register_promise(
                    condition=f"avoid:{pos}",
                    behavior=f"Never visit {pos}",
                    expiry=1000,
                    penalty="cost:40.0"
                )
            
            self.print_step(f"✓ Registered {len(forbidden_positions)} promises")
            self.print_step(f"  Forbidden positions: {forbidden_positions}")
            
            world = GridWorld(
                width=12, height=12,
                start_pos=(0, 5),
                goal_pos=(11, 6),
                danger_tiles=set(),
                forbidden_tiles=set(forbidden_positions)
            )
            world.reset()
            
            self.print_step(f"✓ World: {world._start_pos} → {world.target_pos}")
            self.print_step(f"  Note: Forbidden positions are on/near direct path!")
            
            observation = world.get_observation()
            visited = []
            
            self.print_step("\nNavigation starting...")
            
            for step in range(150):
                result = agent.step(observation)
                world_result = world.step(result['action'])
                observation = world_result['observation']
                visited.append(observation['agent_pos'])
                
                if world.check_goal_reached(observation):
                    break
            
            violations = sum(1 for pos in visited if pos in forbidden_positions)
            
            result = {
                'scenario': 'promise_keeping',
                'promises_registered': len(forbidden_positions),
                'promise_violations': violations,
                'steps_taken': step + 1,
                'goal_reached': world.check_goal_reached(observation),
                'status': 'SUCCESS - Promises kept' if violations == 0 else 'FAILURE - Promises broken'
            }
            
            self.print_result(result)
            self.results.append(result)
            
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
    
    def demo_energy_crisis(self):
        """
        Demonstration 3: Decision Making Under Energy Crisis
        
        Shows how the agent makes critical decisions when
        energy levels are critically low.
        """
        self.print_header("Decision Making Under Energy Crisis")
        
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        
        try:
            agent = Agent(enable_journal_llm=False, db_path=temp_db.name)
            
            world = GridWorld(
                width=10, height=10,
                start_pos=(0, 0),
                goal_pos=(9, 9),
                danger_tiles=set(),
                forbidden_tiles=set()
            )
            world.reset()
            
            # Create energy crisis
            observation = world.get_observation()
            observation['energy'] = 0.25
            
            self.print_step("⚠️  ENERGY CRISIS!")
            self.print_step(f"  Starting energy: {observation['energy']:.2f}")
            self.print_step(f"  Distance to goal: {abs(9-0) + abs(9-0)} steps")
            self.print_step(f"  Can the agent make it?")
            
            energy_log = [observation['energy']]
            
            self.print_step("\nCrisis navigation starting...")
            
            for step in range(100):
                result = agent.step(observation)
                world_result = world.step(result['action'])
                observation = world_result['observation']
                energy_log.append(observation['energy'])
                
                if step % 15 == 0:
                    self.print_step(f"Step {step:2d}: Energy {observation['energy']:.3f}")
                
                if observation['energy'] <= 0:
                    self.print_step("⚠️  Energy depleted!")
                    break
                
                if world.check_goal_reached(observation):
                    self.print_step("✓ Goal reached!")
                    break
            
            result = {
                'scenario': 'energy_crisis',
                'initial_energy': energy_log[0],
                'final_energy': observation['energy'],
                'min_energy': min(energy_log),
                'goal_reached': world.check_goal_reached(observation),
                'survived': observation['energy'] > 0,
                'steps_taken': step + 1
            }
            
            self.print_result(result)
            self.results.append(result)
            
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
    
    def demo_adaptive_behavior(self):
        """
        Demonstration 4: Adaptive Behavior to Perturbations
        
        Shows how the agent adapts when goals change
        mid-simulation.
        """
        self.print_header("Adaptive Behavior to Goal Perturbations")
        
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        
        try:
            agent = Agent(enable_journal_llm=False, db_path=temp_db.name)
            
            world = GridWorld(
                width=12, height=12,
                start_pos=(0, 0),
                goal_pos=(6, 6),
                danger_tiles=set(),
                forbidden_tiles=set()
            )
            world.reset()
            
            self.print_step(f"✓ Initial goal: {world.target_pos}")
            
            observation = world.get_observation()
            goal_changes = []
            
            self.print_step("\nSimulation with perturbations starting...")
            
            for step in range(200):
                # Introduce perturbations
                if step == 30:
                    new_goal = (11, 11)
                    world.target_pos = new_goal
                    observation['goal'] = new_goal
                    goal_changes.append((step, new_goal))
                    self.print_step(f"\n⚡ PERTURBATION: Goal changed to {new_goal}")
                
                if step == 60:
                    new_goal = (11, 0)
                    world.target_pos = new_goal
                    observation['goal'] = new_goal
                    goal_changes.append((step, new_goal))
                    self.print_step(f"\n⚡ PERTURBATION: Goal changed to {new_goal}")
                
                result = agent.step(observation)
                world_result = world.step(result['action'])
                observation = world_result['observation']
                
                if step in [25, 35, 55, 65, 90]:
                    dist = abs(observation['agent_pos'][0] - world.target_pos[0]) + \
                           abs(observation['agent_pos'][1] - world.target_pos[1])
                    self.print_step(f"Step {step:3d}: Pos {observation['agent_pos']}, "
                                   f"Dist to goal: {dist}")
                
                if world.check_goal_reached(observation):
                    self.print_step(f"\n✓ Final goal reached at step {step}!")
                    break
            
            result = {
                'scenario': 'adaptive_behavior',
                'goal_changes': len(goal_changes),
                'final_goal_reached': world.check_goal_reached(observation),
                'total_steps': step + 1,
                'adaptation_success': world.check_goal_reached(observation)
            }
            
            self.print_result(result)
            self.results.append(result)
            
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
    
    def demo_multi_constraint(self):
        """
        Demonstration 5: Multi-Constraint Optimization
        
        Shows agent navigating with multiple competing constraints:
        dangers, promises, energy, and efficiency.
        """
        self.print_header("Multi-Constraint Optimization")
        
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        
        try:
            agent = Agent(enable_journal_llm=False, db_path=temp_db.name)
            
            # Multiple promises
            forbidden = [(4, 4), (5, 5), (6, 6)]
            for pos in forbidden:
                agent.register_promise(
                    condition=f"avoid:{pos}",
                    behavior=f"Avoid {pos}",
                    expiry=1000,
                    penalty="cost:35.0"
                )
            
            self.print_step(f"✓ Registered {len(forbidden)} promises")
            
            # Complex world
            dangers = {(2, 2), (7, 7), (8, 8), (9, 9)}
            world = GridWorld(
                width=12, height=12,
                start_pos=(0, 0),
                goal_pos=(11, 11),
                danger_tiles=dangers,
                forbidden_tiles=set(forbidden)
            )
            world.reset()
            
            self.print_step(f"✓ World with {len(dangers)} danger zones")
            self.print_step("  Agent must navigate maze of constraints!")
            
            observation = world.get_observation()
            violations = {'danger': 0, 'promise': 0}
            
            self.print_step("\nComplex navigation starting...")
            
            for step in range(200):
                result = agent.step(observation)
                world_result = world.step(result['action'])
                observation = world_result['observation']
                
                pos = observation['agent_pos']
                if pos in dangers:
                    violations['danger'] += 1
                if pos in forbidden:
                    violations['promise'] += 1
                
                if step % 30 == 0:
                    self.print_step(f"Step {step:3d}: Violations - "
                                   f"Danger: {violations['danger']}, "
                                   f"Promise: {violations['promise']}")
                
                if world.check_goal_reached(observation):
                    break
            
            result = {
                'scenario': 'multi_constraint',
                'danger_violations': violations['danger'],
                'promise_violations': violations['promise'],
                'total_constraints': len(dangers) + len(forbidden),
                'goal_reached': world.check_goal_reached(observation),
                'steps_taken': step + 1,
                'perfect_run': violations['danger'] == 0 and violations['promise'] == 0
            }
            
            self.print_result(result)
            self.results.append(result)
            
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
    
    def run_all_demos(self):
        """Run all demonstration scenarios."""
        print("\n" + "="*80)
        print("AInception Agent - Comprehensive Demonstration Suite")
        print("="*80)
        
        start_time = time.time()
        
        self.demo_full_lifecycle()
        self.demo_promise_keeping()
        self.demo_energy_crisis()
        self.demo_adaptive_behavior()
        self.demo_multi_constraint()
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "="*80)
        print("DEMONSTRATION SUITE COMPLETE")
        print("="*80)
        print(f"\nRan {len(self.results)} demonstrations in {total_time:.2f}s")
        print("\nResults Summary:")
        for i, result in enumerate(self.results, 1):
            print(f"  {i}. {result['scenario']}: ", end="")
            if 'status' in result:
                print(result['status'])
            elif result.get('goal_reached', False):
                print("✓ SUCCESS")
            else:
                print("⚠ INCOMPLETE")
        
        # Save results
        report_path = '/tmp/demo_results.json'
        with open(report_path, 'w') as f:
            json.dump({
                'demonstrations': self.results,
                'total_time': total_time,
                'timestamp': time.time()
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {report_path}")
        print("="*80 + "\n")
    
    def save_results(self, filepath: str):
        """Save demonstration results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)


def main():
    """Main entry point for demo script."""
    parser = argparse.ArgumentParser(
        description='AInception Agent Comprehensive Demonstrations'
    )
    parser.add_argument(
        '--scenario',
        choices=['full', 'promise', 'crisis', 'adaptive', 'multi'],
        help='Run specific demonstration scenario'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all demonstration scenarios'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    parser.add_argument(
        '--output',
        default='/tmp/demo_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    runner = DemoRunner(verbose=not args.quiet)
    
    if args.all:
        runner.run_all_demos()
    elif args.scenario:
        scenario_map = {
            'full': runner.demo_full_lifecycle,
            'promise': runner.demo_promise_keeping,
            'crisis': runner.demo_energy_crisis,
            'adaptive': runner.demo_adaptive_behavior,
            'multi': runner.demo_multi_constraint
        }
        scenario_map[args.scenario]()
        runner.save_results(args.output)
    else:
        # Default: run all
        runner.run_all_demos()


if __name__ == '__main__':
    main()

# Tutorial: Advanced Planning with ML

This tutorial explores AInception's advanced ML-powered planning capabilities, including diffusion models, neural-augmented A*, and LLM goal decomposition.

## Prerequisites

- Complete the [Basic Setup](basic_setup.md) tutorial
- NVIDIA GPU recommended for optimal performance
- Basic understanding of machine learning concepts

## Overview

AInception's planning system combines classical algorithms with cutting-edge ML:
- **A* Search**: Optimal pathfinding with heuristics
- **Diffusion Models**: Creative, non-greedy trajectory generation
- **Neural Networks**: Learned value functions and heuristics
- **LLM Integration**: Goal decomposition and narrative reasoning

## Step 1: Enable ML Features

First, let's create an agent with full ML capabilities:

```python
#!/usr/bin/env python3
"""
Advanced ML Planning Demo
Showcases diffusion planning, neural augmentation, and LLM integration.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.core import Agent
from worlds.gridworld import GridWorld
from viz.diffusion_planner import SimpleTrajectoryDiffusion
from viz.llm_module import LLMDecomposer

def main():
    print("🚀 Advanced ML Planning Demo")
    
    # Initialize agent with full ML features
    agent = Agent(
        enable_journal_llm=True,
        enable_neural_planner=True,
        enable_diffusion=True
    )
    
    # Create a more complex world
    world = GridWorld(
        width=12,
        height=12,
        goal_pos=(10, 10),
        danger_tiles={(4, 4), (4, 5), (5, 4), (5, 5), (7, 8), (8, 7)},
        resource_tiles={(3, 3), (9, 2)},  # Energy restoration points
        warm_zones={(1, 1), (11, 11)},    # Temperature regulation
        slip_chance=0.15
    )
    
    # Initialize ML components
    diffusion_planner = SimpleTrajectoryDiffusion(grid_size=12, max_length=30)
    llm_decomposer = LLMDecomposer(model_name="gpt2")
    
    # Set complex goal
    complex_goal = "Navigate to the target while maintaining energy above 0.4 and avoiding all danger zones"
    
    print(f"Complex Goal: {complex_goal}")
    
    # Step 1: LLM Goal Decomposition
    print("\n🧠 Step 1: LLM Goal Decomposition")
    observation = world.reset()
    
    subgoals = llm_decomposer.decompose_goal(
        complex_goal,
        context={
            "current_pos": observation['agent_pos'],
            "goal_pos": observation['goal'],
            "energy": observation['energy'],
            "danger_tiles": list(observation['danger_tiles'])
        }
    )
    
    print("Decomposed subgoals:")
    for i, subgoal in enumerate(subgoals, 1):
        print(f"  {i}. {subgoal}")
    
    # Step 2: Diffusion-Based Path Generation
    print("\n🌊 Step 2: Diffusion-Based Path Generation")
    
    # Generate multiple creative paths
    start_pos = observation['agent_pos']
    goal_pos = observation['goal']
    
    diffusion_paths = diffusion_planner.sample(
        start=start_pos,
        goal=goal_pos,
        num_samples=5,
        context={
            "danger_tiles": observation['danger_tiles'],
            "world_size": (12, 12)
        }
    )
    
    print(f"Generated {len(diffusion_paths)} diffusion paths:")
    for i, path in enumerate(diffusion_paths):
        print(f"  Path {i+1}: Length {len(path)}, Score {path.get('score', 'N/A')}")
    
    # Visualize paths
    visualize_paths(world, diffusion_paths, start_pos, goal_pos)
    
    # Step 3: Neural-Augmented Planning
    print("\n🧠 Step 3: Neural-Augmented Planning")
    
    # Run agent with neural planning
    step_count = 0
    max_steps = 100
    done = False
    neural_path = []
    
    while not done and step_count < max_steps:
        result = agent.step(observation)
        observation, reward, done, info = world.step(result['action'])
        
        neural_path.append(observation['agent_pos'])
        step_count += 1
        
        # Print every 10 steps to avoid clutter
        if step_count % 10 == 0:
            print(f"Step {step_count}: Position {observation['agent_pos']}, Energy {observation['energy']:.2f}")
        
        if observation['agent_pos'] == observation['goal']:
            done = True
            break
    
    print(f"Neural planning complete: {step_count} steps")
    
    # Step 4: Compare Planning Methods
    print("\n📊 Step 4: Planning Method Comparison")
    
    # Classical A* (baseline)
    classical_result = run_classical_planning(world, start_pos, goal_pos)
    
    # Hybrid approach (diffusion + classical)
    hybrid_result = run_hybrid_planning(world, start_pos, goal_pos, diffusion_paths)
    
    # Print comparison
    print("Planning Method Comparison:")
    print(f"  Classical A*: {classical_result['steps']} steps, {classical_result['success']}")
    print(f"  Diffusion:    {len(diffusion_paths[0])} steps (best path)")
    print(f"  Neural:       {step_count} steps, {done}")
    print(f"  Hybrid:       {hybrid_result['steps']} steps, {hybrid_result['success']}")
    
    # Step 5: Performance Analysis
    print("\n⚡ Step 5: Performance Analysis")
    analyze_performance([
        ("Classical", classical_result),
        ("Neural", {"steps": step_count, "success": done}),
        ("Hybrid", hybrid_result)
    ])

def visualize_paths(world, paths, start, goal):
    """Visualize different planning paths."""
    print("Generating path visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw world
    world_map = np.zeros((world.height, world.width))
    
    # Mark danger tiles
    for x, y in world.danger_tiles:
        world_map[y, x] = -1
    
    # Mark resource tiles
    for x, y in getattr(world, 'resource_tiles', set()):
        world_map[y, x] = 0.5
    
    # Draw base map
    ax.imshow(world_map, cmap='RdYlGn', alpha=0.7)
    
    # Draw paths
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, path in enumerate(paths[:5]):
        path_coords = np.array(path)
        ax.plot(path_coords[:, 0], path_coords[:, 1], 
                color=colors[i % len(colors)], 
                linewidth=2, alpha=0.8, 
                label=f'Path {i+1}')
    
    # Mark start and goal
    ax.plot(start[0], start[1], 'go', markersize=12, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal')
    
    ax.set_title('Diffusion-Generated Paths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('planning_paths.png', dpi=150, bbox_inches='tight')
    print("Path visualization saved as 'planning_paths.png'")

def run_classical_planning(world, start, goal):
    """Run classical A* planning."""
    # This is a simplified version - in practice, use the actual planner
    from agent.policy.planner import simple_astar
    
    try:
        path = simple_astar(
            start=start,
            goal=goal,
            world_size=(world.width, world.height),
            obstacles=world.danger_tiles
        )
        return {"steps": len(path), "success": len(path) > 0}
    except:
        return {"steps": float('inf'), "success": False}

def run_hybrid_planning(world, start, goal, diffusion_paths):
    """Combine diffusion creativity with classical optimization."""
    if not diffusion_paths:
        return {"steps": float('inf'), "success": False}
    
    # Take the best diffusion path and optimize it
    best_path = min(diffusion_paths, key=len)
    
    # In a real implementation, you would:
    # 1. Use diffusion path as initialization
    # 2. Apply local optimization (e.g., smoothing)
    # 3. Verify safety constraints
    
    return {"steps": len(best_path), "success": True}

def analyze_performance(results):
    """Analyze and compare planning performance."""
    print("Performance Metrics:")
    
    for method, result in results:
        steps = result.get('steps', float('inf'))
        success = result.get('success', False)
        
        if success and steps != float('inf'):
            efficiency = 1.0 / steps  # Simple efficiency metric
            print(f"  {method:10s}: ✓ {steps:3d} steps (efficiency: {efficiency:.3f})")
        else:
            print(f"  {method:10s}: ✗ Failed")

if __name__ == "__main__":
    main()
```

Save this as `advanced_planning_demo.py` and run it:

```bash
python advanced_planning_demo.py
```

## Step 2: Understanding Diffusion Planning

Diffusion models generate creative, non-greedy paths by learning from diverse trajectory data.

### How It Works
1. **Training**: Learn from many example paths (successful and failed)
2. **Sampling**: Generate new paths by denoising random trajectories  
3. **Conditioning**: Guide generation with start/goal positions and constraints
4. **Evaluation**: Score paths based on safety, efficiency, and creativity

### Advantages
- **Exploration**: Finds novel paths classical algorithms might miss
- **Robustness**: Handles complex obstacle patterns
- **Adaptability**: Learns from experience and human feedback

### Configuration
```python
# Configure diffusion model
diffusion_config = {
    "model_size": "small",      # small, medium, large
    "timesteps": 1000,          # Denoising steps
    "beta_schedule": "cosine",  # Noise schedule
    "guidance_scale": 7.5       # Conditioning strength
}

diffusion = SimpleTrajectoryDiffusion(**diffusion_config)
```

## Step 3: Neural-Augmented A*

Classical A* search enhanced with learned heuristics and value functions.

### Neural Components
1. **Value Network**: Estimates cost-to-go from any position
2. **Heuristic Network**: Learned admissible heuristics
3. **Risk Predictor**: Estimates danger and constraint violations

### Training Process
```python
# Example neural training loop
from viz.neural_planner import NeuralPlannerTrainer

trainer = NeuralPlannerTrainer()

for episode in range(1000):
    # Generate training data from agent experience
    trajectory = agent.run_episode(world)
    
    # Extract state-value pairs
    training_data = extract_training_data(trajectory)
    
    # Update neural networks
    trainer.update(training_data)
    
    if episode % 100 == 0:
        print(f"Episode {episode}: Loss = {trainer.get_loss()}")
```

### Using Neural Planner
```python
# Enable neural augmentation
agent = Agent(
    enable_neural_planner=True,
    neural_config={
        "confidence_threshold": 0.7,  # Use neural when confident
        "fallback_to_classical": True, # Safety fallback
        "update_online": True          # Continue learning
    }
)
```

## Step 4: LLM Goal Decomposition

Large Language Models break complex goals into manageable subgoals.

### Goal Types
- **Spatial**: "Navigate to position X via waypoint Y"
- **Temporal**: "Maintain energy above 0.5 for 50 steps"
- **Conditional**: "If temperature drops, find warm zone"
- **Social**: "Avoid promises violations while achieving goal"

### Example Decomposition
```python
complex_goal = """
Navigate to the treasure while:
1. Maintaining energy above 0.4
2. Avoiding all danger zones  
3. Collecting the key first
4. Not breaking promise to stay away from sacred grounds
"""

llm = LLMDecomposer(model_name="gpt-3.5-turbo")
subgoals = llm.decompose_goal(complex_goal, context=current_state)

# Output:
# 1. Navigate to key location safely
# 2. Collect key while monitoring energy
# 3. Find energy restoration if needed  
# 4. Plan path to treasure avoiding sacred grounds
# 5. Execute final approach to treasure
```

## Step 5: Performance Optimization

### GPU Acceleration
```python
# Enable CUDA for faster inference
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion_planner.to(device)
neural_planner.to(device)
```

### Model Quantization
```python
# Reduce model size with quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

llm = LLMDecomposer(
    model_name="gpt2",
    quantization_config=quantization_config
)
```

### Caching and Batching
```python
# Cache frequent computations
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_neural_heuristic(state_hash):
    return neural_planner.compute_heuristic(state_hash)

# Batch diffusion sampling
paths = diffusion_planner.sample_batch(
    starts=[(0,0), (1,1), (2,2)],
    goals=[(5,5), (6,6), (7,7)],
    batch_size=3
)
```

## Step 6: Evaluation and Debugging

### Planning Metrics
```python
def evaluate_planner(planner, test_scenarios):
    metrics = {
        "success_rate": 0,
        "avg_path_length": 0,
        "avg_computation_time": 0,
        "path_optimality": 0
    }
    
    for scenario in test_scenarios:
        start_time = time.time()
        path = planner.plan(scenario['start'], scenario['goal'])
        end_time = time.time()
        
        # Update metrics
        if path and reaches_goal(path, scenario['goal']):
            metrics["success_rate"] += 1
        
        metrics["avg_path_length"] += len(path) if path else float('inf')
        metrics["avg_computation_time"] += end_time - start_time
        
        # Compare to optimal path length
        optimal_length = scenario.get('optimal_length')
        if optimal_length and path:
            metrics["path_optimality"] += len(path) / optimal_length
    
    # Normalize by number of scenarios
    for key in metrics:
        metrics[key] /= len(test_scenarios)
    
    return metrics
```

### Visualization Tools
```python
# Visualize planning process
def visualize_planning_process(world, agent):
    """Create animated visualization of planning."""
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots()
    
    def animate(frame):
        ax.clear()
        
        # Draw world state at frame
        draw_world(ax, world, frame)
        
        # Draw agent's current considerations
        if hasattr(agent, 'planning_debug'):
            draw_search_tree(ax, agent.planning_debug)
        
        ax.set_title(f'Planning Process - Step {frame}')
    
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=100)
    anim.save('planning_process.gif', writer='pillow')
```

## Step 7: Advanced Techniques

### Multi-Objective Planning
```python
# Balance multiple objectives
from agent.policy.multi_objective import MultiObjectivePlanner

planner = MultiObjectivePlanner(
    objectives=[
        ("path_length", 1.0),      # Minimize distance
        ("energy_cost", 2.0),      # Conserve energy  
        ("risk", 5.0),             # Avoid danger
        ("promise_violations", 10.0) # Keep commitments
    ]
)

path = planner.plan_pareto_optimal(start, goal, context)
```

### Hierarchical Planning
```python
# Plan at multiple abstraction levels
high_level_plan = plan_abstract_actions(start, goal)  # "go north", "find energy"
detailed_plan = []

for abstract_action in high_level_plan:
    # Refine each abstract action into concrete steps
    concrete_steps = refine_action(abstract_action, current_state)
    detailed_plan.extend(concrete_steps)
```

### Online Learning
```python
# Continuously improve from experience
class AdaptivePlanner:
    def __init__(self):
        self.experience_buffer = []
        self.neural_net = PlanningNetwork()
    
    def plan(self, start, goal):
        # Use current knowledge
        path = self.neural_net.plan(start, goal)
        return path
    
    def learn_from_experience(self, trajectory, reward):
        # Update neural network from results
        self.experience_buffer.append((trajectory, reward))
        
        if len(self.experience_buffer) >= 32:  # Batch size
            batch = self.experience_buffer[-32:]
            self.neural_net.train_on_batch(batch)
```

## Troubleshooting

### Common Issues

**Slow Performance**
- Enable GPU acceleration
- Reduce model sizes
- Use batch processing
- Cache frequent computations

**Poor Path Quality**
- Increase diffusion timesteps
- Adjust guidance scale
- Retrain with more diverse data
- Combine with classical planning

**High Memory Usage**  
- Use model quantization
- Clear intermediate results
- Reduce batch sizes
- Enable gradient checkpointing

### Debugging Tools
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Profile performance
import cProfile
cProfile.run('agent.step(observation)', 'planning_profile.prof')

# Visualize neural network decisions
from viz.neural_analysis import plot_attention_maps
plot_attention_maps(neural_planner, test_state)
```

## Next Steps

Now that you understand advanced planning, explore:
- [Multi-Agent Social Interactions](multi_agent.md) 
- [RLHF Training and Alignment](rlhf_training.md)
- [Custom World Environments](custom_worlds.md)

## Resources

- [Diffusion Models Paper](https://arxiv.org/abs/2006.11239)
- [Neural A* Algorithm](https://arxiv.org/abs/2009.07476)  
- [LLM Planning Survey](https://arxiv.org/abs/2305.14992)

---

**Congratulations!** You've mastered AInception's advanced ML planning capabilities. These techniques enable agents to plan creatively, efficiently, and safely in complex environments. 🧠✨
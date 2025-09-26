# Tutorial: Basic Setup and First Agent

Welcome to AInception! This tutorial will guide you through setting up your development environment and running your first autonomous agent with constitutional AI and homeostatic drives.

## Prerequisites

Before you begin, ensure you have:
- Python 3.10 or higher
- Git (for cloning the repository)
- At least 8GB of RAM
- Optional: NVIDIA GPU with CUDA 11.8+ for ML features

## Step 1: Installation

### Clone the Repository

```bash
git clone https://github.com/Steake/AInception.git
cd AInception
```

### Set Up Virtual Environment

It's highly recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch 2.1+ (ML framework)
- PyQt6 (GUI framework)
- Transformers (LLM integration)
- Diffusers (diffusion models)
- And many more...

### Verify Installation

Run the test suite to ensure everything is working:

```bash
python run_tests.py --all
```

You should see output indicating all tests pass. If you encounter any failures, check your Python version and dependencies.

## Step 2: Your First Agent

Let's create a simple script to run an agent in a gridworld environment.

Create a file called `my_first_agent.py`:

```python
#!/usr/bin/env python3
"""
My First AInception Agent
A simple example showing basic agent usage.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.core import Agent
from worlds.gridworld import GridWorld

def main():
    print("🚀 Starting AInception Agent Demo")
    
    # Initialize the agent
    print("Initializing agent...")
    agent = Agent(enable_journal_llm=False)  # Disable LLM for simplicity
    
    # Create a simple gridworld
    print("Creating gridworld...")
    world = GridWorld(
        width=8,
        height=8,
        goal_pos=(7, 7),  # Goal in bottom-right corner
        danger_tiles={(3, 3), (4, 4)},  # Some danger tiles to avoid
        slip_chance=0.1  # 10% chance of slipping
    )
    
    # Reset world to initial state
    observation = world.reset()
    done = False
    step_count = 0
    max_steps = 100
    
    print(f"Agent starts at position: {observation['agent_pos']}")
    print(f"Goal position: {observation['goal']}")
    print("Starting simulation...\n")
    
    # Run simulation
    while not done and step_count < max_steps:
        # Get agent's decision
        result = agent.step(observation)
        
        # Execute action in world
        observation, reward, done, info = world.step(result['action'])
        
        step_count += 1
        
        # Print progress
        print(f"Step {step_count:3d}: Action={result['action']}, Position={observation['agent_pos']}")
        print(f"         Reason: {result['justification']}")
        print(f"         Energy: {observation['energy']:.2f}, Temperature: {observation['temperature']:.2f}")
        print()
        
        # Check if goal reached
        if observation['agent_pos'] == observation['goal']:
            print("🎉 Goal reached!")
            done = True
            break
    
    if step_count >= max_steps:
        print("⏰ Maximum steps reached")
    
    # Print final summary
    print(f"\nSimulation complete!")
    print(f"Steps taken: {step_count}")
    print(f"Final position: {observation['agent_pos']}")
    print(f"Goal position: {observation['goal']}")
    print(f"Success: {'Yes' if observation['agent_pos'] == observation['goal'] else 'No'}")

if __name__ == "__main__":
    main()
```

Run your first agent:

```bash
python my_first_agent.py
```

You should see output showing the agent navigating toward the goal while managing its drives!

## Step 3: Understanding the Output

Let's break down what you're seeing:

### Agent Decision Process
Each step shows:
- **Action**: The agent's chosen action (move, pickup, drop, etc.)
- **Position**: Current (x, y) coordinates
- **Reason**: The agent's justification for its decision
- **Energy/Temperature**: Current homeostatic drive values

### Key Concepts
- **Homeostatic Drives**: The agent tries to maintain energy and temperature near optimal setpoints
- **Constitutional Principles**: The agent follows ethical rules (avoid harm, keep promises)
- **Justification**: Every decision includes reasoning, making the agent interpretable

## Step 4: Visualization GUI

Now let's run the interactive visualization:

```bash
python viz/main.py
```

This launches a PyQt6 GUI where you can:
- Watch the agent navigate in real-time
- See drive states as colored bars
- View thought bubbles with reasoning
- Experiment with ML features (diffusion planning, LLM decomposition)

### GUI Controls
- **Play/Pause**: Control simulation
- **Step**: Single-step through decisions
- **Reset**: Return to initial state
- **Speed Slider**: Adjust simulation speed
- **ML Toggle**: Enable/disable ML features

## Step 5: Running Test Scenarios

AInception includes predefined test scenarios to validate agent behavior:

```bash
# Run Day 1 acceptance tests
python cli.py test --day 1 --episodes 5

# Run Day 2 perturbation tests
python cli.py test --day 2 --perturbations

# Generate a comprehensive report
python cli.py report --output test_results.json
```

These tests verify:
- Basic navigation and goal achievement
- Constitutional principle adherence
- Promise keeping under stress
- Drive management

## Step 6: Configuration

You can customize agent behavior through YAML configuration files:

### Drives Configuration (`config/drives.yaml`)
```yaml
drives:
  energy:
    setpoint: 0.7    # Optimal energy level
    weight: 1.0      # Importance weight
    initial: 0.65    # Starting value
    decay_rate: 0.01 # Energy decay per step
  temperature:
    setpoint: 0.5
    weight: 0.8
    initial: 0.55
```

### Principles Configuration (`config/principles.yaml`)
```yaml
principles:
  - name: "do_not_harm"
    initial_rank: 1
    description: "Never cause harm to self or others"
  - name: "keep_promises"
    initial_rank: 2
    description: "Honor all registered commitments"
  - name: "conserve_energy"
    initial_rank: 3
    description: "Maintain energy above critical threshold"
```

Try modifying these values and rerunning your agent to see how behavior changes!

## Step 7: Adding ML Features

Enable advanced ML features by modifying your agent initialization:

```python
# Enable all ML features
agent = Agent(
    enable_journal_llm=True,    # LLM for narrative generation
    enable_neural_planner=True, # Neural-augmented planning
    enable_diffusion=True       # Diffusion-based path generation
)
```

**Note**: ML features require more computational resources and may need GPU acceleration.

## Common Issues and Solutions

### Import Errors
If you get import errors:
```bash
# Make sure you're in the project directory
cd AInception

# Check your Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Performance Issues
For better performance:
- Enable GPU acceleration if available
- Reduce world size for testing
- Disable ML features for faster simulation

### GUI Issues
If the visualization doesn't start:
- Ensure PyQt6 is properly installed
- Check your display settings
- Try running without GUI: `python my_first_agent.py`

## Next Steps

Now that you have a basic agent running, explore these advanced tutorials:
- [Advanced Planning with ML](advanced_planning.md)
- [Multi-Agent Social Interactions](multi_agent.md)
- [Custom World Environments](custom_worlds.md)
- [RLHF Training and Alignment](rlhf_training.md)

## Getting Help

If you run into issues:
1. Check the [API documentation](../API.md)
2. Look at the test files in `tests/` for more examples
3. Join our Discord community for real-time help
4. Open an issue on GitHub

---

**Congratulations!** 🎉 You've successfully set up AInception and run your first constitutional AI agent. The agent you just created demonstrates the key principles of autonomous behavior with ethical constraints and homeostatic drives.

Happy experimenting! 🚀✨
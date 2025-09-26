# AInception API Reference

This document provides comprehensive documentation for all AInception modules, classes, and functions. Use this as your guide for integrating with and extending the AInception framework.

## Table of Contents

- [Core Agent](#core-agent)
- [Drive System](#drive-system)
- [Constitution](#constitution)
- [Social Promises](#social-promises)
- [Planning & Imagination](#planning--imagination)
- [Visualization](#visualization)
- [World Environments](#world-environments)
- [ML Components](#ml-components)
- [Database](#database)
- [CLI Interface](#cli-interface)

---

## Core Agent

### `Agent` Class

The central orchestrator that coordinates all agent subsystems.

```python
from agent.core import Agent

# Initialize agent
agent = Agent(
    enable_journal_llm=False,  # Optional LLM integration
    config_dir="config/",      # Configuration directory
    db_path="agent_state.db"   # Database path
)
```

#### Methods

##### `step(observation: Dict[str, Any]) -> Dict[str, Any]`

Execute one agent decision step.

**Parameters:**
- `observation`: Environment state dict containing:
  - `agent_pos`: Tuple[int, int] - Agent position (x, y)
  - `goal`: Tuple[int, int] - Target position (optional)
  - `energy`: float - Current energy level [0,1]
  - `temperature`: float - Current temperature [0,1]
  - `social_proximity`: float - Social drive state [0,1]
  - `danger_tiles`: Set[Tuple[int, int]] - Dangerous positions
  - `forbidden_tiles`: Set[Tuple[int, int]] - Forbidden positions

**Returns:**
Dictionary with:
- `action`: Dict - Action to execute
- `justification`: str - Reasoning for the decision
- `logs`: Dict - Additional logging information

**Example:**
```python
observation = {
    "agent_pos": (2, 3),
    "goal": (7, 7),
    "energy": 0.6,
    "temperature": 0.5,
    "social_proximity": 0.3
}

result = agent.step(observation)
print(f"Action: {result['action']}")
print(f"Reasoning: {result['justification']}")
```

##### `register_promise(promise_text: str, tiles: Set[Tuple[int, int]]) -> str`

Register a social promise to avoid certain tiles.

**Returns:** Promise ID string

##### `get_drive_summary() -> Dict[str, float]`

Get current state of all homeostatic drives.

##### `set_goal(position: Tuple[int, int]) -> None`

Set agent's goal position.

---

## Drive System

### `DriveSystem` Class

Manages homeostatic needs with quadratic cost functions.

```python
from agent.drives import DriveSystem, build_from_config

# Build from YAML config
drive_system = build_from_config("config/drives.yaml")

# Manual construction
drive_system = DriveSystem()
drive_system.add_drive("energy", setpoint=0.7, weight=1.0, current=0.6)
```

#### Methods

##### `update(observations: Dict[str, float]) -> None`

Update drive values from environment observations.

##### `get_cost() -> float`

Calculate total quadratic cost across all drives.

##### `get_reward() -> float`

Get reward signal (negative cost).

##### `project_delta(hypothetical_changes: Dict[str, float]) -> float`

Estimate cost change from hypothetical drive modifications.

### `Drive` Class

Individual drive representation.

**Attributes:**
- `name`: str - Drive identifier
- `setpoint`: float - Optimal value
- `weight`: float - Importance weight
- `current`: float - Current value
- `min_val`: float - Minimum allowed value (default: 0.0)
- `max_val`: float - Maximum allowed value (default: 1.0)
- `decay_rate`: float - Passive decay per tick (default: 0.0)

---

## Constitution

### `Constitution` Class

Manages ethical principles with rankings and violation detection.

```python
from agent.constitution import Constitution

constitution = Constitution()
constitution.load_from_file("config/principles.yaml")
```

#### Methods

##### `evaluate(action_context: Dict[str, Any]) -> Dict[str, Any]`

Check action against constitutional principles.

**Returns:**
- `violations`: List[str] - Violated principle names
- `checked`: List[str] - Principles that were evaluated

##### `top(n: int) -> List[Dict[str, Any]]`

Get top N highest-priority principles.

##### `set_ranking(new_order: List[str], proof: Dict[str, Any]) -> bool`

Re-rank principles with justification proof.

**Proof format:**
```python
proof = {
    "reason": "Explanation for re-ranking",
    "tradeoffs": ["list of compromises made"],
    "timestamp": int(time.time()),
    "evidence": "Supporting evidence",
    "affected_principles": ["principle1", "principle2"]
}
```

---

## Social Promises

### `PromiseBook` Class

Tracks and enforces social commitments.

```python
from agent.social import PromiseBook

promise_book = PromiseBook()
```

#### Methods

##### `register(promise_text: str, context: Dict[str, Any]) -> str`

Register a new promise.

**Returns:** Unique promise ID

##### `check_violations(planned_action: Dict[str, Any]) -> List[str]`

Check if an action would violate any promises.

##### `get_active_promises() -> List[Dict[str, Any]]`

Get all currently active promises.

---

## Planning & Imagination

### `Planner` Class

A*-based pathfinding with ML augmentation.

```python
from agent.policy.planner import Planner, PlannerConfig

config = PlannerConfig(
    max_depth=50,
    cost_weights={
        "step": 1.0,
        "drive": 2.0,
        "risk": 5.0
    }
)

planner = Planner(config)
```

#### Methods

##### `plan(start: Tuple[int, int], goal: Tuple[int, int], context: Dict[str, Any]) -> Dict[str, Any]`

Generate plan from start to goal.

**Returns:**
- `path`: List[Tuple[int, int]] - Planned positions
- `actions`: List[Dict] - Action sequence
- `cost`: float - Total plan cost
- `justification`: str - Reasoning

### `Imagination` Class

Model predictive control for future state rollouts.

```python
from agent.imagination import Imagination

imagination = Imagination(rollout_depth=5)
```

#### Methods

##### `rollout(current_state: Dict[str, Any], candidate_actions: List[Dict]) -> List[Dict]`

Simulate future states for action candidates.

---

## Visualization

### `MainWindow` Class

PyQt6-based visualization interface.

```python
from viz.main import MainWindow
from PyQt6.QtWidgets import QApplication

app = QApplication([])
window = MainWindow()
window.show()
app.exec()
```

### `AInceptionAdapter` Class

Bridge between agent and visualization.

```python
from viz.adapter import AInceptionAdapter

adapter = AInceptionAdapter(agent)
state = adapter.get_state()
```

---

## World Environments

### `GridWorld` Class

2D grid environment for agent navigation tasks.

```python
from worlds.gridworld import GridWorld

world = GridWorld(
    width=8,
    height=8,
    goal_pos=(7, 7),
    danger_tiles={(3, 3), (4, 4)},
    slip_chance=0.1
)
```

#### Methods

##### `reset() -> Dict[str, Any]`

Reset environment to initial state.

##### `step(action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]`

Execute action in environment.

**Returns:** `(observation, reward, done, info)`

##### `get_observation() -> Dict[str, Any]`

Get current environment state.

### `ArmEnv` Class

2-DOF robotic arm environment.

```python
from worlds.arm import ArmEnv

arm = ArmEnv(
    joint_limits=[(-180, 180), (-90, 90)],
    target_pos=(0.5, 0.3)
)
```

---

## ML Components

### `SimpleTrajectoryDiffusion` Class

Diffusion model for creative path planning.

```python
from viz.diffusion_planner import SimpleTrajectoryDiffusion

diffusion = SimpleTrajectoryDiffusion(
    grid_size=8,
    max_length=20
)

# Generate trajectory
trajectory = diffusion.sample(
    start=(0, 0),
    goal=(7, 7),
    num_samples=5
)
```

### `LLMDecomposer` Class

Large language model integration for goal decomposition.

```python
from viz.llm_module import LLMDecomposer

llm = LLMDecomposer(model_name="gpt2")
subgoals = llm.decompose_goal(
    "Navigate to the target while avoiding danger",
    context={"current_pos": (2, 3), "goal": (7, 7)}
)
```

### `RLHFTrainer` Class

Reinforcement Learning from Human Feedback.

```python
from viz.rlhf_module import RLHFTrainer

trainer = RLHFTrainer(
    base_model_path="./models/base_policy",
    learning_rate=3e-4
)

trainer.train_on_feedback(
    trajectories=trajectories,
    feedback_scores=scores
)
```

---

## Database

### `DatabaseManager` Class

SQLite-based persistence layer.

```python
from database import DatabaseManager

db = DatabaseManager("agent_state.db")
db.initialize()
```

#### Methods

##### `log_decision(agent_id: str, tick: int, action: Dict, justification: str) -> None`

Log agent decision to database.

##### `get_agent_summary(agent_id: str) -> Dict[str, Any]`

Get performance summary for an agent.

##### `query_decisions(filters: Dict[str, Any]) -> List[Dict]`

Query decision history with filters.

---

## CLI Interface

### Command Line Usage

```bash
# Train agent
python cli.py train --day 1 --episodes 5

# Run tests
python cli.py test --day 1 --with-promises
python cli.py test --day 2 --perturbations

# Generate reports
python cli.py report --output results.json

# Interactive demo
python cli.py demo --world gridworld --interactive
```

### `TestRunner` Class

Programmatic test execution.

```python
from cli import TestRunner

runner = TestRunner()
results = runner.run_day_1_tests(episodes=5)
```

---

## Configuration

### YAML Configuration Files

#### `config/drives.yaml`
```yaml
drives:
  energy:
    setpoint: 0.7
    weight: 1.0
    initial: 0.65
    decay_rate: 0.01
  temperature:
    setpoint: 0.5
    weight: 0.8
    initial: 0.55
```

#### `config/principles.yaml`
```yaml
principles:
  - name: "do_not_harm"
    initial_rank: 1
    description: "Never cause harm to self or others"
  - name: "keep_promises"
    initial_rank: 2
    description: "Honor all registered commitments"
```

---

## Error Handling

### Common Exceptions

- `DriveError`: Issues with drive system
- `ConstitutionViolation`: Principle violations
- `PromiseConflict`: Promise violations
- `PlanningError`: Planning failures

### Example Error Handling

```python
try:
    result = agent.step(observation)
except ConstitutionViolation as e:
    print(f"Action violates principle: {e.principle_name}")
    # Handle violation
except PlanningError as e:
    print(f"Planning failed: {e.message}")
    # Use fallback behavior
```

---

## Performance Considerations

### Optimization Guidelines

1. **Agent Steps**: Target <100ms per decision
2. **Visualization**: Maintain 60+ FPS
3. **ML Inference**: <2s for LLM responses
4. **Memory**: Reasonable usage for consumer hardware

### Profiling

```python
import cProfile

# Profile agent step
cProfile.run('agent.step(observation)')
```

---

## Extension Points

### Custom Drives

```python
from agent.drives import Drive

class CustomDrive(Drive):
    def update_custom_logic(self, observation):
        # Custom update logic
        pass
```

### Custom Worlds

```python
from worlds.base import BaseWorld

class MyWorld(BaseWorld):
    def step(self, action):
        # Custom environment logic
        pass
```

### ML Plugins

```python
from viz.base import MLModule

class MyMLModule(MLModule):
    def process(self, input_data):
        # Custom ML processing
        pass
```

---

## Best Practices

### Code Style
- Follow PEP 8 guidelines
- Use type hints consistently
- Add comprehensive docstrings
- Write unit tests for all components

### Performance
- Profile critical paths
- Use appropriate data structures
- Leverage GPU when available
- Implement graceful CPU fallbacks

### Testing
- Test both success and failure cases
- Mock external dependencies
- Use reproducible random seeds
- Validate against known scenarios

---

For more examples and advanced usage patterns, see the [tutorials](tutorials/) directory and examine the test cases in the `tests/` directory.
# Implementation Guide for Situated Autonomous Agent

## Project Structure
```
autonomous_agent/
├── agent/
│   ├── __init__.py
│   ├── core.py              # Main agent class with step() function
│   ├── drives.py            # Homeostatic drives module
│   ├── perception.py        # Perception and event extraction
│   ├── journal.py           # Self-journal with GPT-2 integration
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── reflex.py        # Fast reflex layer
│   │   └── planner.py       # Deliberative planner (MPC/A*)
│   ├── constitution.py      # Principle management
│   ├── imagination.py       # Rollout simulator
│   └── social.py           # Promise and commitment management
├── worlds/
│   ├── __init__.py
│   ├── gridworld.py        # 2D grid world implementation
│   └── arm.py              # 2-DOF arm environment
├── database/
│   ├── __init__.py
│   ├── schema.py           # SQLite schema definitions
│   └── models.py           # Database access layer
├── config/
│   ├── drives.yaml         # Drive setpoints and weights
│   ├── principles.yaml     # Initial constitution
│   └── world_config.yaml   # World parameters
├── tests/
│   ├── unit/              # Unit tests for each module
│   ├── integration/       # Full system tests
│   └── scenarios/         # Day 1 and Day 2 test scenarios
├── cli.py                 # Command-line interface
├── requirements.txt       # Dependencies
└── README.md             # Documentation

```

## Key Implementation Details

### 1. Drives Module
```python
class DriveSystem:
    def __init__(self):
        self.drives = {
            'energy': {'setpoint': 0.7, 'current': 0.5, 'weight': 1.0},
            'temperature': {'setpoint': 0.5, 'current': 0.5, 'weight': 0.8},
            'social_proximity': {'setpoint': 0.3, 'current': 0.0, 'weight': 0.5}
        }
    
    def compute_error(self):
        # Return weighted sum of squared errors
        pass
```

### 2. Event Structure
```python
Event = {
    'timestamp': int,
    'type': str,  # 'state_change', 'goal_contact', 'harm', 'help', etc.
    'delta': dict,  # What changed
    'cause': str,  # Attributed cause
    'consequence': str,  # Expected consequence
    'surprise_level': float  # Deviation from expected
}
```

### 3. Constitution Proof Object
```python
class TradeoffProof:
    def __init__(self, old_ranking, new_ranking, justification):
        self.old_ranking = old_ranking
        self.new_ranking = new_ranking
        self.justification = justification
        self.cost_analysis = {}  # What was sacrificed
        self.expected_benefit = {}  # What is gained
```

### 4. Promise Structure
```python
class Promise:
    def __init__(self, promise_id, condition, behavior, expiry, penalty_fn):
        self.id = promise_id
        self.condition = condition
        self.expected_behavior = behavior
        self.expiry = expiry
        self.penalty_function = penalty_fn
        self.status = 'active'  # 'active', 'fulfilled', 'broken'
```

### 5. Justification Object
```python
class ActionJustification:
    def __init__(self, action, goal, principles_checked, tradeoffs):
        self.action = action
        self.chosen_goal = goal
        self.principles_checked = principles_checked[:3]  # Top 3
        self.expected_tradeoffs = tradeoffs
        self.promise_compliance = []
```

## Database Schema

```sql
-- Events table
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER,
    event_type TEXT,
    delta JSON,
    cause TEXT,
    consequence TEXT,
    surprise_level REAL
);

-- Journal entries
CREATE TABLE journal (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER,
    event_id INTEGER,
    natural_text TEXT,
    summary_type TEXT,  -- 'immediate', 'hourly', 'daily', 'lesson'
    FOREIGN KEY (event_id) REFERENCES events(id)
);

-- Principles
CREATE TABLE principles (
    id INTEGER PRIMARY KEY,
    principle TEXT,
    rank INTEGER,
    created_at INTEGER,
    modified_at INTEGER
);

-- Principle changes
CREATE TABLE principle_changes (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER,
    old_ranking JSON,
    new_ranking JSON,
    proof JSON
);

-- Promises
CREATE TABLE promises (
    id TEXT PRIMARY KEY,
    condition TEXT,
    expected_behavior TEXT,
    expiry INTEGER,
    penalty_function TEXT,
    status TEXT,
    created_at INTEGER,
    resolved_at INTEGER
);

-- Action logs
CREATE TABLE action_logs (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER,
    action JSON,
    justification JSON,
    drive_errors JSON,
    principles_checked JSON
);
```

## Testing Protocol

### Day 1 Tests
1. **Baseline Performance**
   - 5 episodes per task
   - Record: drive errors, violations, success rates, journal content

2. **Promise Test**
   - Register: "Avoid forbidden zone despite shortcuts"
   - Inject 2 temptations
   - Expect: Refusal with justification

### Day 2 Tests
1. **Environmental Perturbations**
   - Shift goal positions
   - Change friction/temperature
   - Paraphrase temptations

2. **Success Criteria**
   - ≥ Day 1 success rates
   - Zero principle violations
   - Maintained promises
   - Evidence of learning from surprises

## CLI Commands

```bash
# Train agent on Day 1
python cli.py train --day 1 --episodes 5

# Run Day 1 tests
python cli.py test --day 1 --with-promises

# Run Day 2 perturbation tests
python cli.py test --day 2 --perturb

# Export logs
python cli.py export --format json --output logs/

# Generate report
python cli.py report --day all --output report.md
```

## Configuration Examples

### drives.yaml
```yaml
drives:
  energy:
    setpoint: 0.7
    weight: 1.0
    decay_rate: 0.01  # per tick
  temperature:
    setpoint: 0.5
    weight: 0.8
    comfort_range: [0.3, 0.7]
  social_proximity:
    setpoint: 0.3
    weight: 0.5
    max_distance: 5
```

### principles.yaml
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
  - name: "explore_learn"
    initial_rank: 4
    description: "Seek new information when safe"
```

## Minimal Dependencies

```
# requirements.txt
numpy==1.24.0
transformers==4.30.0  # For GPT-2
torch==2.0.0  # Required by transformers
pyyaml==6.0
```

Note: SQLite3 is included in Python standard library.
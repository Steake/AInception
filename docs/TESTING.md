# Testing Documentation

This document provides detailed information about the AInception testing framework, including examples, output screenshots, and best practices.

## Quick Start

```bash
# Run all tests
python run_tests.py --all

# Run specific test category
python run_tests.py --unit          # Unit tests only
python run_tests.py --integration   # Integration tests only
python run_tests.py --scenarios     # Scenario tests only

# Run BDD tests
pytest tests/bdd/ -v

# Run with coverage
python run_tests.py --coverage
```

## Test Categories Overview

### 1. Unit Tests (21 tests)

Unit tests validate individual components in isolation. These tests are fast (< 0.01s) and provide immediate feedback during development.

**Command:**
```bash
python run_tests.py --unit --verbose
```

**Expected Output:**
```
test_drive_errors (tests.test_drives.TestDriveSystem.test_drive_errors)
Test drive error calculation. ... ok
test_drive_initialization (tests.test_drives.TestDriveSystem.test_drive_initialization)
Test drives initialize to target values. ... ok
test_energy_update (tests.test_drives.TestDriveSystem.test_energy_update)
Test energy updates with consumption. ... ok
test_projection_utility (tests.test_drives.TestDriveSystem.test_projection_utility)
Test drive projection utility calculation. ... ok
test_temperature_update (tests.test_drives.TestDriveSystem.test_temperature_update)
Test temperature updates. ... ok
test_principle_evaluation (tests.test_constitution.TestConstitution.test_principle_evaluation)
Test principle evaluation on nodes. ... ok
test_principle_loading (tests.test_constitution.TestConstitution.test_principle_loading)
Test principles load from config. ... ok
test_proof_validation (tests.test_constitution.TestConstitution.test_proof_validation)
Test proof-gated re-ranking. ... ok
test_ranking_system (tests.test_constitution.TestConstitution.test_ranking_system)
Test principle ranking system. ... ok
test_avoid_condition_parsing (tests.test_social.TestPromiseBook.test_avoid_condition_parsing)
Test avoid condition parsing and violation detection. ... ok
test_breach_detection (tests.test_social.TestPromiseBook.test_breach_detection)
Test automatic breach detection. ... ok
test_expiry_handling (tests.test_social.TestPromiseBook.test_expiry_handling)
Test automatic promise expiry. ... ok
test_penalty_calculation (tests.test_social.TestPromiseBook.test_penalty_calculation)
Test penalty value extraction. ... ok
test_promise_lifecycle (tests.test_social.TestPromiseBook.test_promise_lifecycle)
Test promise lifecycle management. ... ok
test_promise_registration (tests.test_social.TestPromiseBook.test_promise_registration)
Test promise registration and structure. ... ok
test_serialization (tests.test_social.TestPromiseBook.test_serialization)
Test promise book serialization. ... ok
test_different_horizons (tests.test_imagination.TestImagination.test_different_horizons)
Test rollout with different horizon lengths. ... ok
test_drive_projection (tests.test_imagination.TestImagination.test_drive_projection)
Test drive state projection accuracy. ... ok
test_multi_step_rollout (tests.test_imagination.TestImagination.test_multi_step_rollout)
Test multi-step action sequence rollout. ... ok
test_risk_assessment (tests.test_imagination.TestImagination.test_risk_assessment)
Test risk score calculation. ... ok
test_single_step_rollout (tests.test_imagination.TestImagination.test_single_step_rollout)
Test single step action rollout. ... ok

----------------------------------------------------------------------
Ran 21 tests in 0.002s

OK

Running AInception Agent Tests...
==================================================
Tests run: 21
Failures: 0
Errors: 0
Success rate: 100.0%

✅ All tests passed!
```

#### Unit Test Coverage by Component

| Component | Tests | Description |
|-----------|-------|-------------|
| **Drive System** | 5 | Homeostatic drives: initialization, updates, error calculation, projection utility |
| **Constitution** | 4 | Principle loading, evaluation, ranking system, proof validation |
| **Promise Book** | 7 | Registration, lifecycle, breach detection, expiry, penalties, serialization |
| **Imagination** | 5 | Single/multi-step rollouts, drive projection, risk assessment, horizons |

### 2. BDD Tests (9 scenarios)

BDD (Behavior-Driven Development) tests use human-readable Gherkin syntax to describe agent behavior. These tests serve as both executable specifications and living documentation.

**Command:**
```bash
pytest tests/bdd/ -v
```

**Expected Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/runner/work/AInception/AInception
plugins: bdd-8.1.0
collecting ... collected 9 items

tests/bdd/step_defs/test_drive_steps.py::test_agent_maintains_energy_levels PASSED [ 11%]
tests/bdd/step_defs/test_drive_steps.py::test_agent_balances_multiple_drives PASSED [ 22%]
tests/bdd/step_defs/test_drive_steps.py::test_agent_responds_to_drive_urgency PASSED [ 33%]
tests/bdd/step_defs/test_navigation_steps.py::test_agent_reaches_goal_without_obstacles PASSED [ 44%]
tests/bdd/step_defs/test_navigation_steps.py::test_agent_navigates_around_danger_tiles PASSED [ 55%]
tests/bdd/step_defs/test_navigation_steps.py::test_agent_maintains_energy_while_navigating PASSED [ 66%]
tests/bdd/step_defs/test_promise_steps.py::test_agent_resists_shortcut_temptation PASSED [ 77%]
tests/bdd/step_defs/test_promise_steps.py::test_agent_sacrifices_efficiency_for_principles PASSED [ 88%]
tests/bdd/step_defs/test_promise_steps.py::test_agent_maintains_promise_under_time_pressure PASSED [100%]

============================== 9 passed in 2.30s ===============================
```

#### BDD Test Coverage by Feature

| Feature | Scenarios | Description |
|---------|-----------|-------------|
| **Agent Navigation** | 3 | Goal reaching, danger avoidance, energy management during navigation |
| **Promise Keeping** | 3 | Resisting shortcuts, sacrificing efficiency, maintaining promises under pressure |
| **Drive Management** | 3 | Energy maintenance, multi-drive balancing, urgency response |

#### Example BDD Feature: Navigation

**File:** `tests/bdd/features/agent_navigation.feature`

```gherkin
Feature: Agent Goal Navigation
  As an AI agent
  I want to navigate to goal positions
  So that I can complete my objectives while maintaining my drives

  Scenario: Agent reaches goal without obstacles
    Given the agent starts at position (0, 0)
    And the goal is at position (7, 7)
    And there are no obstacles
    When the agent navigates for up to 100 steps
    Then the agent should have zero principle violations
    And the agent should make progress toward the goal
```

**Test Output:**
```bash
$ pytest tests/bdd/step_defs/test_navigation_steps.py -v

tests/bdd/step_defs/test_navigation_steps.py::test_agent_reaches_goal_without_obstacles PASSED [ 33%]
tests/bdd/step_defs/test_navigation_steps.py::test_agent_navigates_around_danger_tiles PASSED [ 66%]
tests/bdd/step_defs/test_navigation_steps.py::test_agent_maintains_energy_while_navigating PASSED [100%]

============================== 3 passed in 0.75s ===============================
```

### 3. Integration Tests

Integration tests validate full agent-environment interactions over multiple steps.

**Command:**
```bash
python run_tests.py --integration --verbose
```

Tests include:
- Multi-step planning with drive dynamics
- Principle enforcement during complex scenarios
- Environment interaction loops
- State persistence and recovery

### 4. Scenario Tests

Scenario tests validate specific acceptance criteria for agent behavior.

**Command:**
```bash
python run_tests.py --scenarios --verbose
```

Tests include:
- Day 1 baseline: Basic goal reaching without violations
- Day 1 promise temptation: Resisting shortcuts despite efficiency costs
- Day 2 perturbations: Goal shifts with maintained promises
- Drive sacrifice: Principle adherence over drive optimization

### 5. End-to-End (E2E) Tests

E2E tests provide comprehensive demonstrations of the agent's full capabilities in realistic scenarios.

**Command:**
```bash
pytest tests/test_e2e.py -v -s
```

**Test Categories:**

#### Full Demo Scenarios
Complete agent lifecycle demonstrations:
- `test_full_agent_lifecycle_demo`: Initialization through goal achievement
- `test_promise_enforcement_demo`: Maintaining commitments under temptation

#### Interesting Use Cases
Complex multi-step scenarios:
- `test_energy_crisis_decision_making`: Critical decisions with low resources
- `test_multi_constraint_optimization`: Navigating multiple competing constraints
- `test_adaptive_behavior_to_perturbations`: Responding to dynamic goal changes

#### Performance Metrics
- `test_performance_baseline`: Efficiency and decision speed measurements

**Expected Output:**
```
================================================================================
DEMO: Full Agent Lifecycle
================================================================================
✓ Agent initialized with homeostatic drives and constitutional principles
✓ World created: 10x10 grid from (0, 0) to (9, 9)
✓ Danger zones at: {(3, 3), (5, 5), (7, 7)}

Starting simulation...
  Step 0: Position (1, 0), Energy 0.68, Action: move
  Step 20: Position (5, 1), Energy 0.52, Action: move
  Step 40: Position (7, 4), Energy 0.38, Action: move

✓ Goal reached at step 52!

--------------------------------------------------------------------------------
RESULTS:
  Steps taken: 52
  Initial energy: 0.70
  Final energy: 0.31
  Energy consumed: 0.39
  Goal reached: True
  Path length: 53 positions
--------------------------------------------------------------------------------

6 passed in 1.43s
```

**Interactive Demonstrations:**

Run standalone demo scenarios:
```bash
# Run all demonstrations
python demo_e2e.py --all

# Run specific scenario
python demo_e2e.py --scenario full      # Full lifecycle demo
python demo_e2e.py --scenario promise   # Promise keeping demo
python demo_e2e.py --scenario crisis    # Energy crisis demo
python demo_e2e.py --scenario adaptive  # Adaptive behavior demo
python demo_e2e.py --scenario multi     # Multi-constraint demo

# Save results to file
python demo_e2e.py --all --output results.json
```

**Demo Output Example:**
```
DEMO: Promise Keeping Under Temptation
================================================================================
✓ Registered 1 promise: Avoid position (5, 5)
  Promise ID: 1
  Penalty for violation: 50.0
✓ World: Straight path from (0, 5) to (10, 5)
  Shortcut at (5, 5) is on the direct path!

Navigation starting...
  Agent path: [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (4, 6), ...]
  Visited 18 unique positions
  Promise violated: False
  Steps to goal: 20

--------------------------------------------------------------------------------
✓ SUCCESS: Agent maintained promise despite efficiency cost
--------------------------------------------------------------------------------
```

## Test Execution Time

| Test Suite | Tests | Average Time | Coverage |
|------------|-------|--------------|----------|
| Unit Tests | 21 | ~0.002s | Components |
| BDD Tests | 9 | ~2.3s | Behaviors |
| Integration Tests | Variable | ~1-2s | Workflows |
| Scenario Tests | 4 | ~1.5s | Acceptance |
| **E2E Tests** | **6** | **~1.4s** | **Full Stack Demos** |
| **Total** | **40+** | **~7s** | **Complete System** |

## Coverage Report

To generate a detailed coverage report:

```bash
python run_tests.py --coverage
```

This will output:
- Line-by-line coverage for each module
- Percentage coverage per file
- Missing lines report
- Overall coverage statistics

## Writing New Tests

### Unit Test Example

```python
import unittest
from agent.drives import DriveSystem

class TestDriveSystem(unittest.TestCase):
    def test_energy_update(self):
        """Test energy updates with consumption."""
        spec = {"energy": {"setpoint": 0.7, "weight": 1.0, "initial": 0.5}}
        drives = DriveSystem(spec)
        
        drives.ingest_observation({"energy": 0.6})
        self.assertAlmostEqual(drives.drives["energy"].current, 0.6)
```

### BDD Test Example

1. **Create Feature File** (`tests/bdd/features/my_feature.feature`):

```gherkin
Feature: My Agent Behavior
  Scenario: Agent does something
    Given some initial state
    When some action occurs
    Then expected outcome happens
```

2. **Create Step Definitions** (`tests/bdd/step_defs/test_my_steps.py`):

```python
from pytest_bdd import scenarios, given, when, then

scenarios('../features/my_feature.feature')

@given("some initial state")
def initial_state(context):
    context['state'] = "initialized"

@then("expected outcome happens")
def verify_outcome(context):
    assert context['state'] == "initialized"
```

## Continuous Integration

All tests run automatically on GitHub Actions for every push and pull request:

[![CI Status](https://github.com/Steake/AInception/actions/workflows/ci.yml/badge.svg)](https://github.com/Steake/AInception/actions/workflows/ci.yml)

**CI Pipeline:**
- Tests against Python 3.10, 3.11, 3.12
- Runs all test categories
- Caches dependencies (~30s build time)
- Generates coverage reports
- Reports failures immediately

## Troubleshooting

### Test Failures

If tests fail:

1. **Run specific test with verbose output:**
   ```bash
   python run_tests.py --unit --verbose
   ```

2. **Check specific test:**
   ```bash
   python -m pytest tests/test_drives.py::TestDriveSystem::test_energy_update -v
   ```

3. **View coverage for failed module:**
   ```bash
   python run_tests.py --coverage
   ```

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Ensure you're in the project root: `cd /path/to/AInception` |
| Missing dependencies | Run `pip install -r requirements.txt` |
| Database locked | Delete `*.db` files in project root |
| Slow tests | Use `--failfast` flag to stop on first failure |

## Best Practices

1. **Run tests frequently** during development
2. **Write tests first** for new features (TDD)
3. **Use BDD for behaviors** that need stakeholder review
4. **Keep tests isolated** - no shared state between tests
5. **Mock external dependencies** (APIs, file I/O)
6. **Aim for >90% coverage** on core components
7. **Update tests** when changing behavior

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-bdd documentation](https://pytest-bdd.readthedocs.io/)
- [Gherkin syntax reference](https://cucumber.io/docs/gherkin/reference/)
- [CONTRIBUTING.md](CONTRIBUTING.md) - Testing guidelines
- [tests/bdd/README.md](tests/bdd/README.md) - BDD test guide

---

For questions or issues with tests, please open a GitHub issue or check the [Contributing Guidelines](CONTRIBUTING.md).

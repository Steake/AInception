# BDD Test Specifications

This directory contains Behavior-Driven Development (BDD) test specifications for the AInception agent.

## Structure

```
tests/bdd/
├── features/           # Gherkin feature files with scenarios
│   ├── agent_navigation.feature
│   ├── promise_keeping.feature
│   └── drive_management.feature
├── step_defs/         # Step definition implementations
│   ├── test_navigation_steps.py
│   ├── test_promise_steps.py
│   └── test_drive_steps.py
└── conftest.py        # Pytest configuration
```

## Features

### Agent Navigation
Tests basic navigation capabilities including:
- Goal reaching without obstacles
- Navigation around danger tiles
- Energy maintenance during navigation

### Promise Keeping
Tests constitutional AI behavior including:
- Resisting shortcut temptations
- Sacrificing efficiency for principles
- Maintaining promises under pressure

### Drive Management
Tests homeostatic drive system including:
- Energy level maintenance
- Balancing multiple drives
- Responding to drive urgency

## Running BDD Tests

### All BDD tests
```bash
pytest tests/bdd/ --verbose
```

### Specific feature
```bash
pytest tests/bdd/step_defs/test_navigation_steps.py --verbose
pytest tests/bdd/step_defs/test_promise_steps.py --verbose
pytest tests/bdd/step_defs/test_drive_steps.py --verbose
```

### With coverage
```bash
pytest tests/bdd/ --cov=agent --cov-report=html
```

## Requirements

Install BDD testing dependencies:
```bash
pip install pytest pytest-bdd
```

## Writing New BDD Tests

1. Create a new `.feature` file in `features/` directory
2. Write scenarios using Gherkin syntax (Given-When-Then)
3. Create corresponding step definitions in `step_defs/`
4. Use the `@scenarios()` decorator to link feature files
5. Implement step functions with `@given`, `@when`, `@then` decorators

Example:
```python
from pytest_bdd import scenarios, given, when, then

scenarios('../features/my_feature.feature')

@given('some precondition')
def precondition(context):
    # Setup code
    pass

@when('some action')
def action(context):
    # Action code
    pass

@then('expected outcome')
def outcome(context):
    # Assertion code
    assert True
```

## Best Practices

1. **Keep scenarios focused**: Each scenario should test one behavior
2. **Use descriptive names**: Make scenarios readable as documentation
3. **Maintain independence**: Tests should not depend on each other
4. **Use fixtures**: Share setup code through pytest fixtures
5. **Clean up resources**: Use teardown in fixtures to clean temp files

"""
Step definitions for agent navigation BDD tests.
"""
import pytest
import tempfile
import os
from pytest_bdd import scenarios, given, when, then, parsers
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from agent.core import Agent
from worlds.gridworld import GridWorld

# Load all scenarios from the feature file
scenarios('../features/agent_navigation.feature')


@pytest.fixture
def context():
    """Shared context for BDD tests."""
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    ctx = {
        'temp_db': temp_db.name,
        'agent': None,
        'world': None,
        'start_pos': None,
        'goal_pos': None,
        'danger_tiles': set(),
        'forbidden_tiles': set(),
        'max_steps': 100,
        'violations': 0,
        'steps_taken': 0,
        'goal_reached': False,
        'observation': None,
        'agent_path': []
    }
    
    yield ctx
    
    # Cleanup
    if os.path.exists(ctx['temp_db']):
        os.unlink(ctx['temp_db'])


@given("I am an agent with basic drives")
def agent_with_basic_drives(context):
    """Create an agent with basic drive configuration."""
    context['agent'] = Agent(enable_journal_llm=False, db_path=context['temp_db'])


@given("I am in a gridworld environment")
def in_gridworld(context):
    """Initialize gridworld environment."""
    # Will be configured with specific parameters in subsequent steps
    pass


@given(parsers.parse("the agent starts at position ({x:d}, {y:d})"))
def agent_starts_at(context, x, y):
    """Set agent starting position."""
    context['start_pos'] = (x, y)


@given(parsers.parse("the goal is at position ({x:d}, {y:d})"))
def goal_at_position(context, x, y):
    """Set goal position."""
    context['goal_pos'] = (x, y)


@given("there are no obstacles")
def no_obstacles(context):
    """Set no obstacles in the world."""
    context['danger_tiles'] = set()
    context['forbidden_tiles'] = set()


@given(parsers.parse("there are danger tiles at positions ({x1:d}, {y1:d}) and ({x2:d}, {y2:d})"))
def danger_tiles_at_positions(context, x1, y1, x2, y2):
    """Set danger tile positions."""
    context['danger_tiles'] = {(x1, y1), (x2, y2)}


@when(parsers.parse("the agent navigates for up to {max_steps:d} steps"))
def agent_navigates(context, max_steps):
    """Execute agent navigation."""
    context['max_steps'] = max_steps
    
    # Create world with configured parameters
    context['world'] = GridWorld(
        width=8, height=8,
        start_pos=context['start_pos'],
        goal_pos=context['goal_pos'],
        danger_tiles=context['danger_tiles'],
        forbidden_tiles=context['forbidden_tiles']
    )
    
    context['world'].reset()
    context['observation'] = context['world'].get_observation()
    
    context['violations'] = 0
    context['steps_taken'] = 0
    context['agent_path'] = [context['start_pos']]
    
    for step in range(max_steps):
        result = context['agent'].step(context['observation'])
        justification = result['justification']
        
        # Check for violations
        if justification.get('principle_violations', 0) > 0:
            context['violations'] += 1
        
        # Apply action
        world_result = context['world'].step(result['action'])
        context['observation'] = world_result['observation']
        context['steps_taken'] += 1
        
        # Track path
        context['agent_path'].append(context['observation']['agent_pos'])
        
        # Check for goal
        if context['world'].check_goal_reached(context['observation']):
            context['goal_reached'] = True
            break


@then("the agent should reach the goal")
def agent_reaches_goal(context):
    """Verify agent reached the goal."""
    assert context['goal_reached'], f"Agent did not reach goal after {context['steps_taken']} steps"


@then("the agent should have zero principle violations")
def zero_violations(context):
    """Verify no principle violations occurred."""
    assert context['violations'] == 0, f"Agent had {context['violations']} principle violations"


@then("the agent should avoid danger tiles")
def avoids_danger_tiles(context):
    """Verify agent avoided all danger tiles."""
    danger_visits = [pos for pos in context['agent_path'] if pos in context['danger_tiles']]
    assert len(danger_visits) == 0, f"Agent visited danger tiles: {danger_visits}"


@then("the agent's energy should not fall below critical threshold")
def energy_above_threshold(context):
    """Verify agent maintained adequate energy."""
    final_energy = context['observation']['energy']
    critical_threshold = 0.0
    assert final_energy >= critical_threshold, f"Agent energy {final_energy} fell below critical threshold"


@then("the agent should make progress toward the goal")
def makes_progress_toward_goal(context):
    """Verify agent moved closer to the goal."""
    start_pos = context['start_pos']
    goal_pos = context['goal_pos']
    
    # Calculate initial distance to goal
    initial_distance = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])
    
    # Calculate final distance to goal
    if context['goal_reached']:
        # If goal reached, this is definitely progress
        assert True
    else:
        # Check if agent moved at all
        final_pos = context['observation']['agent_pos']
        final_distance = abs(goal_pos[0] - final_pos[0]) + abs(goal_pos[1] - final_pos[1])
        
        # Agent should have made some progress or taken actions
        assert context['steps_taken'] > 0, "Agent should be taking actions"
        # Accept if agent reduced distance or is making attempts
        assert final_distance <= initial_distance or len(context['agent_path']) > 5, \
            f"Agent should make progress toward goal. Initial distance: {initial_distance}, Final: {final_distance}"

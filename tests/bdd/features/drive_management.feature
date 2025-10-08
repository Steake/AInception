Feature: Homeostatic Drive Management
  As an AI agent with biological-inspired drives
  I want to maintain my internal state within acceptable ranges
  So that I can function effectively

  Background:
    Given I am an agent with homeostatic drives
    And I have energy, temperature, and social proximity drives

  Scenario: Agent maintains energy levels
    Given the agent starts with 1.0 energy
    And the agent is in a gridworld with energy sources
    When the agent operates for 50 steps
    Then the agent's energy should remain above 0.1
    And the agent should seek energy when it drops below 0.5

  Scenario: Agent balances multiple drives
    Given the agent starts with balanced drive states
    And the agent has conflicting drive requirements
    When the agent operates for 100 steps
    Then the agent should maintain all drives within acceptable ranges
    And the agent should prioritize critical drives

  Scenario: Agent responds to drive urgency
    Given the agent has energy at 0.3 (below setpoint)
    And the agent has temperature at 0.8 (above setpoint)
    And there are resources available in the environment
    When the agent operates for 30 steps
    Then the agent should prioritize drive optimization
    And the agent should move toward resources that satisfy urgent drives

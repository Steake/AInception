Feature: Agent Goal Navigation
  As an AI agent
  I want to navigate to goal positions
  So that I can complete my objectives while maintaining my drives

  Background:
    Given I am an agent with basic drives
    And I am in a gridworld environment

  Scenario: Agent reaches goal without obstacles
    Given the agent starts at position (0, 0)
    And the goal is at position (7, 7)
    And there are no obstacles
    When the agent navigates for up to 100 steps
    Then the agent should have zero principle violations
    And the agent should make progress toward the goal

  Scenario: Agent navigates around danger tiles
    Given the agent starts at position (0, 0)
    And the goal is at position (7, 7)
    And there are danger tiles at positions (3, 3) and (5, 5)
    When the agent navigates for up to 100 steps
    Then the agent should have zero principle violations
    And the agent should avoid danger tiles
    And the agent should make progress toward the goal

  Scenario: Agent maintains energy while navigating
    Given the agent starts at position (0, 0)
    And the goal is at position (7, 7)
    And there are no obstacles
    When the agent navigates for up to 100 steps
    Then the agent should make progress toward the goal
    And the agent's energy should not fall below critical threshold

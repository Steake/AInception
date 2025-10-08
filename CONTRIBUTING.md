# Contributing to AInception

Thank you for your interest in contributing to AInception! We're excited to have you join our community of developers working on autonomous AI agents with constitutional AI, homeostatic drives, and cutting-edge ML visualization.

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:
- Python 3.10+
- Git
- NVIDIA GPU (recommended for ML features)
- Basic understanding of AI/ML concepts

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/AInception.git
   cd AInception
   ```
3. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development tools
   ```
5. **Run tests** to ensure everything works:
   ```bash
   python run_tests.py --all
   ```

## 🛠️ How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **🐛 Bug fixes**: Fix issues in core agent logic, ML modules, or visualization
- **✨ New features**: Implement new ML techniques, visualization modes, or agent capabilities
- **📚 Documentation**: Improve guides, API docs, or tutorials
- **🧪 Tests**: Add test cases for better coverage
- **🎨 UI/UX improvements**: Enhance the PyQt6 visualization interface
- **⚡ Performance**: Optimize rendering, ML inference, or agent decision-making
- **🤖 ML enhancements**: Add new diffusion models, LLM integrations, or RLHF techniques

### Contribution Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-idea
   # or
   git checkout -b bugfix/issue-description
   ```

2. **Make your changes** following our coding standards:
   - Follow PEP 8 for Python code style
   - Add docstrings for all public functions/classes
   - Include type hints where appropriate
   - Write comprehensive tests for new features

3. **Test your changes**:
   ```bash
   # Run unit tests
   python run_tests.py --unit
   
   # Run integration tests
   python run_tests.py --integration
   
   # Test specific scenarios
   python cli.py test --day 1 --episodes 5
   
   # Test visualization
   python viz/main.py
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add cool feature: brief description"
   ```
   
   Use clear, descriptive commit messages following conventional commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for adding tests
   - `refactor:` for code refactoring

5. **Push to your fork**:
   ```bash
   git push origin feature/amazing-idea
   ```

6. **Create a Pull Request** on GitHub with:
   - Clear description of changes
   - Screenshots/videos for UI changes
   - Test results
   - Link to related issues

## 📋 Coding Standards

### Python Code Style
- Follow PEP 8
- Use type hints (`from typing import ...`)
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Documentation
- Add docstrings to all public functions:
  ```python
  def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
      """Execute one agent decision step.
      
      Args:
          observation: Environment state including position, drives, etc.
          
      Returns:
          Dictionary with 'action', 'justification', and 'logs' keys.
      """
  ```

### Testing
- Write unit tests for all new functions
- Include integration tests for major features
- Test both success and failure cases
- Mock external dependencies (LLMs, file I/O)

### ML Components
- Document model architectures and training procedures
- Include performance benchmarks
- Provide fallback modes for CPU-only systems
- Use reproducible random seeds for testing

## 🎯 Areas We Need Help

### High Priority
- **Multi-agent social interactions**: Expand promise negotiation and conflict resolution
- **Advanced ML modules**: Implement new diffusion architectures, graph transformers
- **Performance optimization**: GPU acceleration, memory efficiency
- **Mobile/web deployment**: Export to different platforms

### Medium Priority
- **Additional world environments**: 3D spaces, physics simulations
- **Advanced visualization**: VR/AR integration, better animations
- **Ethical AI features**: Bias detection, fairness metrics
- **Developer tools**: Better debugging, profiling utilities

### Good First Issues
Look for issues labeled `good-first-issue` on GitHub, typically involving:
- Documentation improvements
- Small bug fixes
- Test case additions
- Configuration enhancements

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Test individual components
├── integration/    # Test full workflows
├── scenarios/      # Specific agent scenarios
└── bdd/           # Behavior-Driven Development tests
    ├── features/   # Gherkin feature files
    └── step_defs/  # Step definition implementations
```

### Running Tests
```bash
# All tests
python run_tests.py --all

# Specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --scenarios

# BDD tests
pytest tests/bdd/ --verbose

# Specific BDD feature
pytest tests/bdd/step_defs/test_navigation_steps.py

# Verbose output
python run_tests.py --all --verbose
```

### Test Coverage
- Aim for >90% code coverage
- Test both happy path and edge cases
- Mock external dependencies (OpenAI API, file I/O)

### Writing BDD Tests

BDD (Behavior-Driven Development) tests use Gherkin syntax to describe behavior in natural language:

**Feature File Example** (`tests/bdd/features/my_feature.feature`):
```gherkin
Feature: Agent Navigation
  As an AI agent
  I want to navigate to goals
  So that I can complete objectives

  Scenario: Agent reaches goal
    Given the agent starts at position (0, 0)
    And the goal is at position (5, 5)
    When the agent navigates for up to 50 steps
    Then the agent should reach the goal
```

**Step Definitions** (`tests/bdd/step_defs/test_my_steps.py`):
```python
from pytest_bdd import scenarios, given, when, then

scenarios('../features/my_feature.feature')

@given("the agent starts at position (0, 0)")
def agent_at_origin(context):
    context['start_pos'] = (0, 0)

@then("the agent should reach the goal")
def reaches_goal(context):
    assert context['goal_reached']
```

See `tests/bdd/README.md` for detailed BDD testing guidelines.

### Continuous Integration

All PRs automatically run through GitHub Actions CI which:
- Tests against Python 3.10, 3.11, and 3.12
- Runs all test categories (unit, integration, scenarios, BDD)
- Generates coverage reports
- Caches dependencies for faster builds

Check the Actions tab for build status and detailed logs.

## 📝 Documentation Standards

### API Documentation
- Use clear, descriptive docstrings
- Include parameter types and return values
- Provide usage examples
- Document exceptions that may be raised

### README and Guides
- Use markdown with appropriate headers
- Include code examples that actually work
- Add screenshots for UI features
- Keep language accessible but technical

### Tutorials
- Start with prerequisites
- Provide step-by-step instructions
- Include expected output
- Test all examples regularly

## 🚀 ML Enhancement Guidelines

### Adding New ML Models
1. **Create modular interfaces**: Follow existing patterns in `viz/` directory
2. **Provide CPU fallbacks**: Not everyone has GPUs
3. **Include performance metrics**: Inference time, memory usage
4. **Document training procedures**: How to reproduce results

### Visualization Features
1. **Maintain 60+ FPS**: Profile performance impact
2. **Support different screen sizes**: Responsive design
3. **Provide interactive controls**: Let users experiment
4. **Add tooltips and help**: Make features discoverable

## 🤝 Community Guidelines

### Code Reviews
- Be constructive and respectful
- Focus on the code, not the person
- Suggest improvements with examples
- Acknowledge good work

### Communication
- **Discord**: Join our community server for real-time discussion
- **GitHub Issues**: For bugs, feature requests, and planning
- **GitHub Discussions**: For general questions and showcasing work

### Conduct
We follow a code of conduct based on respect, inclusivity, and collaboration:
- Be welcoming to newcomers
- Respect different perspectives and experience levels
- Focus on constructive feedback
- Help others learn and grow

## 📊 Performance Standards

### Code Performance
- Visualization should maintain 60+ FPS
- Agent decision-making should be <100ms per step
- ML inference should be <2s for LLM responses
- Memory usage should be reasonable for consumer hardware

### Test Performance
- Unit tests should complete in <30 seconds
- Integration tests should complete in <5 minutes
- CI/CD pipeline should complete in <10 minutes

## 🏆 Recognition

Contributors will be recognized through:
- **GitHub contributor graphs**
- **Release notes mentions** for significant contributions
- **Hall of Fame** section for major features
- **Social media shoutouts** for innovative work

## 📞 Getting Help

Stuck on something? We're here to help:

1. **Check existing documentation** and issues first
2. **Ask in Discord** for real-time help
3. **Open a GitHub Discussion** for design questions
4. **Create an issue** for bugs or feature requests

## 🗺️ Roadmap

Check our project roadmap in the main README for upcoming features and how you can contribute to each milestone.

---

**Thank you for contributing to AInception!** Together, we're building the future of autonomous AI agents. Every contribution, no matter how small, helps push the boundaries of what's possible in AI research and development.

*Happy coding!* 🚀✨
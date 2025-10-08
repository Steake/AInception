# GitHub Actions CI/CD

This directory contains GitHub Actions workflow files for continuous integration and deployment.

## Workflows

### CI Workflow (`ci.yml`)

The main CI workflow runs on every push and pull request to `main` and `develop` branches.

#### Features

- **Multi-version Python Testing**: Tests against Python 3.10, 3.11, and 3.12
- **Dependency Caching**: Speeds up builds by caching pip packages
- **Comprehensive Test Suite**: Runs unit, integration, scenario, and BDD tests
- **Coverage Reporting**: Generates code coverage reports

#### Jobs

1. **Test Job**
   - Sets up Python environment
   - Installs dependencies from `requirements.txt`
   - Installs testing tools (pytest, pytest-bdd, coverage)
   - Runs all test categories:
     - Unit tests
     - Integration tests
     - Scenario tests
     - BDD tests
   - Generates coverage report

#### Triggering the Workflow

The workflow automatically runs on:
- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop` branches

#### Local Testing

To run the same tests locally:

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-bdd coverage

# Run unit tests
python run_tests.py --unit --verbose

# Run integration tests
python run_tests.py --integration --verbose

# Run scenario tests
python run_tests.py --scenarios --verbose

# Run BDD tests
pytest tests/bdd/ --verbose

# Run with coverage
python run_tests.py --coverage
```

## Configuration

The workflow uses:
- **actions/checkout@v4**: For checking out the repository
- **actions/setup-python@v4**: For setting up Python
- **actions/cache@v3**: For caching dependencies

## Badges

Add this to your README.md to show CI status:

```markdown
![CI Status](https://github.com/Steake/AInception/actions/workflows/ci.yml/badge.svg)
```

## Troubleshooting

### Build Failures

If the CI build fails:

1. Check the Actions tab in GitHub for detailed logs
2. Reproduce the failure locally using the commands above
3. Common issues:
   - Missing dependencies in `requirements.txt`
   - Test failures due to breaking changes
   - Python version incompatibilities

### Performance

The workflow includes caching to improve performance:
- Pip packages are cached based on `requirements.txt` hash
- Cache is automatically updated when dependencies change

## Adding New Workflows

To add new workflows:

1. Create a new `.yml` file in `.github/workflows/`
2. Define the workflow name, triggers, and jobs
3. Test locally before committing
4. Monitor the Actions tab to ensure it runs correctly

## Best Practices

- Keep workflows focused (one primary purpose per workflow)
- Use matrix builds for multi-version testing
- Cache dependencies to speed up builds
- Set appropriate timeouts for long-running jobs
- Use `continue-on-error` for optional steps
- Add clear job and step names for easy debugging

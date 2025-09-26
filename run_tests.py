#!/usr/bin/env python3
"""
Comprehensive test runner for AInception agent.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --scenarios        # Run only scenario tests
    python run_tests.py --coverage         # Run with coverage report
"""

import unittest
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def discover_tests(pattern="test_*.py", start_dir="tests"):
    """Discover and return test suite."""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern=pattern)
    return suite


def discover_specific_tests(test_names):
    """Discover specific test files."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_name in test_names:
        try:
            if test_name.endswith('.py'):
                test_name = test_name[:-3]  # Remove .py extension
            
            # Import the test module
            module_name = f"tests.{test_name}"
            module = __import__(module_name, fromlist=[test_name])
            
            # Add all test cases from the module
            module_suite = loader.loadTestsFromModule(module)
            suite.addTest(module_suite)
        except ImportError as e:
            print(f"Warning: Could not import {test_name}: {e}")
    
    return suite


def run_test_suite(suite, verbosity=2):
    """Run test suite and return results."""
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    return result


def main():
    parser = argparse.ArgumentParser(description="AInception Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--scenarios", action="store_true", help="Run scenario tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    
    if args.coverage:
        try:
            import coverage
            cov = coverage.Coverage()
            cov.start()
        except ImportError:
            print("Coverage module not installed. Install with: pip install coverage")
            sys.exit(1)
    
    # Determine which tests to run
    if args.unit:
        suite = discover_specific_tests([
            "test_drives", "test_constitution", "test_social", "test_imagination"
        ])
    elif args.integration:
        suite = discover_specific_tests(["test_integration"])
    elif args.scenarios:
        suite = discover_specific_tests(["test_scenarios"])
    else:
        # Run all tests
        suite = discover_tests()
    
    print("Running AInception Agent Tests...")
    print("=" * 50)
    
    result = run_test_suite(suite, verbosity)
    
    if args.coverage:
        cov.stop()
        cov.save()
        print("\nCoverage Report:")
        print("-" * 20)
        cov.report()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('Error:')[-1].strip()}")
    
    # Exit with appropriate code
    if result.failures or result.errors:
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
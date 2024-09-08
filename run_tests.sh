#!/bin/bash

# Run tests with coverage
coverage run -m unittest discover

# Generate coverage report
coverage report -m

# Generate HTML coverage report
coverage html

echo "Tests completed. Check htmlcov/index.html for the coverage report."

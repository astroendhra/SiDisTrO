#!/bin/bash
set -e

# Run the Python script with all arguments passed to the container
python ddp_launch.py "$@"
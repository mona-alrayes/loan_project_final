#!/bin/bash
set -e  # Exit on error

# Debug: Show Python version
echo "Python version: $(python --version)"

# Install remaining dependencies
pip install -r requirements.txt

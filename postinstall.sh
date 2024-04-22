#!/bin/bash

# This script is run after installing apt and Python packages

# This import has the side-effect of installing the required R pacakges the
# first time it's run
python -c 'import change_detection.functions'

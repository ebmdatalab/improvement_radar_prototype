#!/bin/bash
unset CDPATH
cd "$( dirname "${BASH_SOURCE[0]}")"
exec ./run-command.sh jupyter nbconvert --to html --execute notebooks/SICBL_improvement_radar.ipynb --no-input --ExecutePreprocessor.timeout=-1

#!/bin/bash

if [ -f "$HOME/pylocal/bin/activate" ] || [ -z "${VIRTUAL_ENV}" ]; then
    echo Activating pylocal virtual environment
    . "$HOME/pylocal/bin/activate"
fi

#export NUMBA_LOG_LEVEL=DEBUG
#export NUMBA_DEBUGINFO=1

echo Starting script

python -OO "$@"


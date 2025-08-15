#!/bin/bash

if [ -f "$HOME/pylocal/bin/activate" ] || [ -z "${VIRTUAL_ENV}" ]; then
    echo Activating pylocal virtual environment
    . "$HOME/pylocal/bin/activate"
fi


echo Starting script

python -OO "$@"

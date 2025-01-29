#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install activation script
bash "$SCRIPT_DIR/env_vars_activation.sh"

# Install deactivation script
bash "$SCRIPT_DIR/env_vars_deactivation.sh"

echo "Environment scripts installation completed"

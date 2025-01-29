#!/bin/bash

# Directory for conda environment scripts
CONDA_ENV_DIR=~/anaconda3/envs/kaggle_api/etc/conda/activate.d

# Create directory if it doesn't exist
mkdir -p $CONDA_ENV_DIR

# Create the environment variables script
cat << 'EOF' > $CONDA_ENV_DIR/env_vars.sh
#!/bin/bash

# Kaggle API credentials
export KAGGLE_USERNAME="mantoha"
export KAGGLE_KEY="0895a99005cb2d27aeea337035438d03"

# API configuration
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"

# Create Kaggle config directory if it doesn't exist
mkdir -p $KAGGLE_CONFIG_DIR

# Set proper permissions
chmod 700 $KAGGLE_CONFIG_DIR

echo "Kaggle API environment variables have been set"
EOF

# Make the script executable
chmod +x $CONDA_ENV_DIR/env_vars.sh

echo "Activation script installed successfully"

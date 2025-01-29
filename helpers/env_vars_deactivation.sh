#!/bin/bash

# Directory for conda environment scripts
CONDA_ENV_DIR=~/anaconda3/envs/kaggle_api/etc/conda/deactivate.d

# Create directory if it doesn't exist
mkdir -p $CONDA_ENV_DIR

# Create the deactivation script
cat << 'EOF' > $CONDA_ENV_DIR/env_vars.sh
#!/bin/bash

# Unset Kaggle API environment variables
unset KAGGLE_USERNAME
unset KAGGLE_KEY
unset KAGGLE_CONFIG_DIR

echo "Kaggle API environment variables have been unset"
EOF

# Make the script executable
chmod +x $CONDA_ENV_DIR/env_vars.sh

echo "Deactivation script installed successfully"

cat << 'EOF' > ~/anaconda3/envs/kaggle_api/etc/conda/deactivate.d/env_vars.sh
#!/bin/bash
unset KAGGLE_USERNAME
unset KAGGLE_KEY
EOF

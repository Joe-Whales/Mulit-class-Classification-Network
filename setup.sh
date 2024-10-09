#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the environment name
ENV_NAME="fruit_classification"

# Check if Miniconda is installed
if ! command -v conda &> /dev/null
then
    echo "Miniconda is not installed. Please install Miniconda and try again."
    exit 1
fi

# Create a new Miniconda environment
echo "Creating new Miniconda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.8 -y

# Activate the environment
echo "Activating environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install dependencies
echo "Installing dependencies"
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install numpy matplotlib scikit-learn seaborn tqdm pyyaml

# Verify installation
echo "Verifying installation"
python -c "import torch; import torchvision; import numpy; import matplotlib; import sklearn; import seaborn; import tqdm; import yaml; print('All packages imported successfully')"

echo "Setup complete. You can now activate the environment with: conda activate $ENV_NAME"
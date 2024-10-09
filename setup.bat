@echo off

:: Define the environment name
set ENV_NAME=fruit_classification

:: Check if Miniconda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Miniconda is not installed. Please install Miniconda and try again.
    exit /b 1
)

:: Create a new Miniconda environment
echo Creating new Miniconda environment: %ENV_NAME%
call conda create -n %ENV_NAME% python=3.8 -y

:: Activate the environment
echo Activating environment: %ENV_NAME%
call conda activate %ENV_NAME%

:: Install dependencies
echo Installing dependencies
call conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
call pip install numpy matplotlib scikit-learn seaborn tqdm pyyaml

:: Verify installation
echo Verifying installation
python -c "import torch; import torchvision; import numpy; import matplotlib; import sklearn; import seaborn; import tqdm; import yaml; print('All packages imported successfully')"

echo Setup complete. You can now activate the environment with: conda activate %ENV_NAME%
pause
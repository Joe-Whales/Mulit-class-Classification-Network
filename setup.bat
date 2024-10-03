@echo off
setlocal enabledelayedexpansion

:: Set environment variables
set CONDA_ENV_NAME=ai_assignment
set PYTHON_VERSION=3.11
set REQUIREMENTS_FILE=requirements.txt

:: Create Miniconda environment
call conda create -n %CONDA_ENV_NAME% python=%PYTHON_VERSION% -y
if %errorlevel% neq 0 (
    echo Failed to create Conda environment.
    exit /b %errorlevel%
)

:: Activate the environment
call conda activate %CONDA_ENV_NAME%
if %errorlevel% neq 0 (
    echo Failed to activate Conda environment.
    exit /b %errorlevel%
)

:: Create requirements.txt file
echo numpy==1.24.3 > %REQUIREMENTS_FILE%
echo pandas==2.0.1 >> %REQUIREMENTS_FILE%
echo scikit-learn==1.2.2 >> %REQUIREMENTS_FILE%
echo matplotlib==3.7.1 >> %REQUIREMENTS_FILE%
echo jupyter==1.0.0 >> %REQUIREMENTS_FILE%

:: Install PyTorch with CUDA support
call pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo Failed to install PyTorch.
    exit /b %errorlevel%
)

:: Install other requirements
call pip install -r %REQUIREMENTS_FILE%
if %errorlevel% neq 0 (
    echo Failed to install requirements.
    exit /b %errorlevel%
)

echo Environment setup complete.
echo To activate the environment, use: conda activate %CONDA_ENV_NAME%
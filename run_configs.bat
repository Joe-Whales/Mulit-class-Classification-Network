@echo off
for %%f in (configs\config_*.yaml) do (
    echo Running configuration: %%f
    python main.py --config %%f
    if errorlevel 1 (
        echo Error occurred while running %%f
        pause
        exit /b %errorlevel%
    )
)
echo All configurations completed successfully.
pause
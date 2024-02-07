@echo off

SET USE_VENV=1
SET REPOSITORIES_DIR="repositories"
SET VENV_DIR=%~dp0%venv

if %1 == T (
    echo Cleaning virtual env and repositories folder. 
    rmdir /s /q repositories >stdout.txt 2>stderr.txt
    rmdir /s /q venv >stdout.txt 2>stderr.txt
)

if ["%2"] == ["F"] (
    SET USE_VENV=0
    echo You chose not to use venv...    
)

if not defined PYTHON (set PYTHON=python)

dir "%VENV_DIR%\Scripts\Python.exe" >stdout.txt 2>stderr.txt
if %ERRORLEVEL% == 0 goto :end

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating python venv dir %VENV_DIR% in using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv %VENV_DIR% >stdout.txt 2>stderr.txt
if not EXIST .\%REPOSITORIES_DIR% mkdir %REPOSITORIES_DIR%

if ["%USE_VENV%"] == ["0"] goto :skip_venv

echo Activating python venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv %PYTHON%
call %VENV_DIR%\Scripts\activate.bat

:skip_venv
echo Cloning LLaVA repository...
cd .\repositories
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
git checkout da83c48cb0679155b82dbf4603229bb7ce4929d4 >stdout.txt 2>stderr.txt
echo Patching LLaVA...
copy /y ..\..\patches\LLaVa\pyproject.toml .
copy /y ..\..\patches\LLaVa\builder.py .\llava\model
echo Installing LLaVA. This could take a few minutes...
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
cd ..\..
echo Installing other requirements. This could take a few minutes...
pip install -r requirements.txt
pip uninstall bitsandbytes --yes
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.0-py3-none-win_amd64.whl
echo Patching PIL...
copy /y .\patches\PIL\Image.py .\venv\Lib\site-packages\PIL
echo Patching Gradio...
copy /y .\patches\Gradio\image.py .\venv\Lib\site-packages\gradio\components
goto:end

if %ERRORLEVEL% == 0 goto :end
echo Cannot activate python venv, aborting... %VENV_DIR%
goto :show_stdout_stderr

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stderr:
type stderr.txt

:end
echo.
echo All done!. Launch 'start.bat'.
pause


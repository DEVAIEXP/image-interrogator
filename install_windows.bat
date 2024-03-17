@echo off

SET USE_VENV=1
SET REPOSITORIES_DIR="repositories"
SET VENV_DIR=%~dp0%venv

if ["%1"] == ["T"] (
    echo Cleaning virtual env and repositories folder. 
    rmdir /s /q repositories >stdout.txt 2>stderr.txt
    rmdir /s /q venv >stdout.txt 2>stderr.txt
    pause
)

if ["%2"] == ["F"] (
    SET USE_VENV=0
    echo You chose not to use venv...    
)

if not defined PYTHON (set PYTHON=python)

dir "%VENV_DIR%\Scripts\Python.exe" >stdout.txt 2>stderr.txt
if %ERRORLEVEL% == 0 goto :end

if ["%USE_VENV%"] == ["0"] goto :skip_venv

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating python venv dir %VENV_DIR% in using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv %VENV_DIR% >stdout.txt 2>stderr.txt

echo Activating python venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv %PYTHON%
call %VENV_DIR%\Scripts\activate.bat

:skip_venv
if not EXIST .\%REPOSITORIES_DIR% mkdir %REPOSITORIES_DIR%
echo Cloning LLaVA repository...
cd .\repositories
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
git checkout da83c48cb0679155b82dbf4603229bb7ce4929d4 >stdout.txt 2>stderr.txt
echo Patching LLaVA...
copy /y ..\..\patches\LLaVa\pyproject.toml .
copy /y ..\..\patches\LLaVa\builder.py .\llava\model
echo Installing LLaVA. This could take a few minutes...
pip3 install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -e .
cd ..\..
echo Installing other requirements. This could take a few minutes...
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip uninstall bitsandbytes --yes
pip install bitsandbytes==0.43.0 --upgrade
pip install https://huggingface.co/elismasilva/wheels/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install auto-gptq==0.7.1 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
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


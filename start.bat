@echo off
SET PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
SET CUDA_VISIBLE_DEVICES=0
SET USE_VENV=1
SET VENV_DIR="%~dp0%venv"


if ["%USE_VENV%"] == ["0"] goto :skip_venv

echo Activating python venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
call %VENV_DIR%\Scripts\activate.bat

:skip_venv
python image-interrogator.py

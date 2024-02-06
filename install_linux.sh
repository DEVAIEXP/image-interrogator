#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"

use_venv=1
delimiter="################################################################"
# config
HOME_DIR=$( getent passwd "$USER" | cut -d: -f6 )
REPOSITORIES_DIR="$(pwd)/repositories"
ENV_DIR="./venv"

if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

# Verify python installation
if [[ "$(${python_cmd} -V)" =~ "Python 3" ]] &> /dev/null
then
    printf "Python 3 is installed" 
    printf "\n%s\n" "${delimiter}"
else
    printf "Python 3 is not installed. Please install Python 3.10.6 before continue.."     
    printf "\n%s\n" "${delimiter}"
    exit 1	
fi

if [[ $use_venv -eq 1 ]] && ! "${python_cmd}" -c "import venv" &>/dev/null
then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: python3-venv is not installed, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

if [[ $use_venv -eq 1 ]] && [[ -z "${VIRTUAL_ENV}" ]];
then    
    if [[ ! -d "${ENV_DIR}" ]]
    then
        printf "\n%s\n" "${delimiter}"
        printf "Creating python venv"
        printf "\n%s\n" "${delimiter}"
        "${python_cmd}" -m venv "${ENV_DIR}"        
    fi
    
    if [[ -f "${ENV_DIR}"/bin/activate ]]
    then        
        printf "Activating python venv"
        printf "\n%s\n" "${delimiter}"
        source venv/bin/activate        
    else
        printf "\n%s\n" "${delimiter}"
        printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
        printf "\n%s\n" "${delimiter}"
        exit 1
    fi
else
    printf "\n%s\n" "${delimiter}"
    printf "python venv already activate or run without venv. ${VIRTUAL_ENV}"
    printf "\n%s\n" "${delimiter}"
fi

if [[ ! -d "${REPOSITORIES_DIR}" ]]
    then
        mkdir -p $REPOSITORIES_DIR
    fi

printf "\n%s\n" "${delimiter}"
printf "Cloning LLaVA repository..."
cd ./repositories
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
git checkout da83c48cb0679155b82dbf4603229bb7ce4929d4 &>/dev/null
printf "\n%s\n" "${delimiter}"
printf "Patching LLaVA..."
cp -f ../../patches/LLaVa/pyproject.toml .
cp -f ../../patches/LLaVa/builder.py ./llava/model
printf "\n%s\n" "${delimiter}"
printf "Installing LLaVA. This could take a few minutes..."
printf "\n%s\n" "${delimiter}"
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
cd ../..
printf "\n%s\n" "${delimiter}"
printf "Installing other requirements. This could take a few minutes..."
pip install -r requirements.txt
printf "Patching PIL..."
cp -f ./patches/PIL/Image.py ./venv/lib64/"$(ls ./venv/lib64)"/site-packages/PIL
printf "\n%s\n" "${delimiter}"
printf "Patching Gradio..."
cp -f ./patches/Gradio/image.py ./venv/lib64/"$(ls ./venv/lib64)"/site-packages/gradio/components
printf "\n%s\n" "${delimiter}"
printf "All done!"
printf "\n%s\n" "${delimiter}"
printf "Launching image-interrogator..."
printf "\n%s\n" "${delimiter}"
./start.sh
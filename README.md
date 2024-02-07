# IMAGE-interrogator

*Want to improve your SOTA image captioning experience for Stable Diffusion? The **IMAGE Interrogator** is here to make this job easier!*


## About

The **IMAGE Interrogator** is a variant of the original [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator) tool that brings all original features and adds other large models like [LLaVa](https://llava-vl.github.io/) and [CogVml](https://github.com/THUDM/CogVLM) for SOTA image captioning. Then you can train with fine-tuning on your datasets or use resulting prompts with text-to-image models like [Stable Diffusion](https://github.com/CompVis/stable-diffusion) on [DreamStudio](https://beta.dreamstudio.ai/) to create cool art!


## Installation

Use Python version 3.10.* and have the Python virtual environment installed. 
Then run the following commands in the terminal:
```bash
git clone https://github.com/DEVAIEXP/image-interrogator.git
cd image-interrogator
(for linux  ) source install_linux.sh
(for windows) install_windows.bat
```
### Aditional parameters for installation scripts:
Assuming **T** for True and **F** for False:
* Passing T on first parameter will force deletion of the Venv and repositories folders for a clean installation.
```bash
(for linux  ) source install_linux.sh T
(for windows) install_windows.bat T
````
* Passing T on second parameter will disable use of Venv during installation.
```bash
(for linux  ) source install_linux.sh F T
(for windows) install_windows.bat F T
````
## Running
**If you ran the installation without Venv, you must edit the script files and change the USE_VENV variable to the value 0 before starting to run the tool.**
```bash
(for linux  ) source start.sh
(for windows) start.bat
```

## Running with customization


The *start.sh* and *start.bat* scripts trigger the image-interrogator.py script via Python. The python script allows you to enter some parameters:
* `--cache-dir`: Folder to download cache models. If specified, models will be fully downloaded to the specified path instead of the HF cache.
* `--share`: Create a public access link
Edit this files and change last line, for e.g:
```bash
(for linux  ) python image-interrogator.py --cache-dir "/mnt/c/models" --share
(for windows) python image-interrogator.py --cache-dir "c:\models" --share
```


Note: The linux script is configured to run on WSL 2. If you are running on a linux installation you will need to adjust the LD_LIBRARY_PATH variable in the file with the correct path of your CUDA Toolkit.

IMAGE Interrogator support **4-bit** quantization and **8-bit** quantization (except for CogVLM and CogAgent only 4-bit quantization is enabled) for low memory usage. Precision type parameters have also been added to the interface such as **FP16** and **BF16**. On systems with low VRAM you can try 4-bit quantization or check `Optimize settings for low VRAM` in Load options from interface. It will reduce the amount of VRAM needed (at the cost of some speed and quality). 

# Interface parameters
## Prompt tab
* `Folder Path`: Image folder path for batch processing.
* `Single Image Upload`: For processing a single image.
### Generate options

The `Generate options` lets you enable/disable resources that may be generated during or after the prompt is generated. 
* `Include image features in the prompt`: Enable the Features tab where it is possible to select OpenCLIP pretrained CLIP models. It will add image analysis result features such as (artists, flavors, media, movements, trends, negative prompt). Note: for negative prompts, the caption model will be ignored.
* `Don't save dataset images`: This will disable copying of the image in the caption output path.
* `Generate individual caption file`: When enabled, this generates individual '.txt' caption files for each image.
* `Generate dataset`: This will compile a dataset into the output path so that it can be loaded into hugging-face datasets or used in model training.
* `Dataset name`: The name for dataset folder in output path.

### Caption tab
In this tab you can choose your preferred model from the list for generating the caption. For some models like LLaVa some additional parameters are available such as: temperature and top p. LLaVa, CogAgent, CogVLM and Kosmos-2 allow you to use question prompts for generation. We automatically suggest question prompts when the template is selected, but you can change the prompt text to ask what you want about an image for the template. Look for more information about how to generate prompts in these models directly on the official page of the chosen model.

### Features tab
For selection of OpenCLIP pretrained CLIP Model. Only one feature mode can selected at a time.

### Action buttons
* `Generate prompt for single image`: The caption/prompt will be generated in the output path for the image selected in the defined settings.
* `Batch process for folder`: The caption/prompt will be generated for all images in the folder selected for the output path with the defined settings.
* `Test all caption models`: A csv file will be generated in the output path, delimited by ';' which will contain the caption/prompt generated in each available caption model for comparison purposes. This does not include results from CLIP Models.
* `Test all CLIP models`: If you chose a winning model in the previous test, you can now test all available CLIP templates for the selected caption template. The result will also be stored in the output path in a csv file delimited by ';'.

## Analyze tab
It returns a list of words in each feature and their scores for the given image, model.

## Others
If you update the version of LLaVa, Gradio and PIL dependencies, this tool will not work correctly. When there is a need to update these dependencies, we will update them in our repository. Whenever there is a new update to this repository, it will be necessary to delete the 'repositories' directory and run the installation script again.
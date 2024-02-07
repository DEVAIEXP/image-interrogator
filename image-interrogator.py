#!/usr/bin/env python3
import argparse
import subprocess
import torch
import csv
import json
import traceback
import datetime
import os, shutil
from typing import List

from util.gui import (
    open_folder)
from clip_interrogator import Config, Interrogator, list_caption_models, list_clip_models
from util.generate import generate_captions
from util.util import memory_cleanup,get_reserved_memory, get_used_memory, get_total_memory, has_win_os


try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed, please install it with 'pip install gradio'")
    exit(1)

folder_symbol = '\U0001f4c2'  # 📂
alert_symbol = '\U0001F6A8' # 🚨

DEVICES= {
    'auto' : 'auto',    
    'cpu' : 'cpu'
}

#Pre-Initilize params
ci = None
config = None

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cache-dir', type=str, default=None, help='Folder to download cache models. If specified, models will be fully downloaded to the specified path instead of the HF cache.')
parser.add_argument('-s', '--share', action='store_true', help='Create a public link')
args = parser.parse_args()

css = ""
if os.path.exists("./style.css"):
        with open(os.path.join("./style.css"), "r", encoding="utf8") as file:            
            css += file.read() + "\n"

if os.path.exists("./.release"):
        with open(os.path.join("./.release"), "r", encoding="utf8") as file:
            release = file.read()
            
memory_cleanup()
# Function to test all models
total_models_count = 0
tested_models_count = 0
session_tested_models_count = 0

def load_ci():
    global ci
    if ci:
        del ci
    memory_cleanup()
    ci = Interrogator(config)
    return ci          

def list_devices() -> List[str]:
    devices = list(DEVICES.keys())
    if torch.cuda.is_available():
        cudas =  list([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        devices = devices + cudas
    return devices

# Function to load tested combinations from CSV
def load_tested_combinations(csv_file):
    combinations = set()
    if os.path.exists(csv_file):
        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                if csv_file.startswith('model_caption'):
                    combinations.add(row[0])  # Caption model name
                else:
                    combinations.add((row[0], row[1]))  # Caption model and CLIP model names
    return combinations

# Function to log to CSV
def log_caption_to_csv(csv_file, caption_model, caption, used_memory):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([caption_model, caption, used_memory])
        
def log_clip_to_csv(csv_file, caption_model, clip_model, caption, used_memory):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([caption_model, clip_model, caption, used_memory])

def set_device(selected):    
    if selected in ('auto', 'cuda'):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device=='cpu':
            print("CUDA is not available, using CPU. Warning: this will be very slow!")
        return device
    else:
        return selected

def prep_test_folder():
    t = datetime.datetime.now()    
    test_path = os.path.join(os.getcwd(),'outputs','tests',f"{t.year}-{t.month:02d}-{t.day:02d}")
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path, exist_ok=True)
    return test_path

def test_all_caption_models(image, load_mode, device):
    try:
        global tested_models_count, session_tested_models_count
        if not image:
            raise Exception("Images must be provided!")
            
        total_models_count = len(list_caption_models())
        test_path = prep_test_folder()
        csv_file = f"{test_path}/model_caption_test_results.csv"
        tested_combinations = load_tested_combinations(csv_file)
        load_config()    
        initial_used_memory = get_used_memory()
        for caption_model in sorted(sorted(list_caption_models(), key=lambda t: t[1]), key=lambda t: t[0], reverse=False):   
            question_prompt_text = get_question_prompt(caption_model)     
            if (caption_model) not in tested_combinations:
                caption = image_to_prompt(image=image, question_prompt=question_prompt_text, generate_features=False, check_dataset=False, dataset_name='Test', feature_mode=None, 
                                        precision_type='FP16', load_mode=load_mode, clip_model_name=None, caption_model_name=caption_model, device=device,
                                        temperature=0.2, top_p=0.7, max_new_tokens=0, check_txt_caption=False, check_lowvram=False, check_only_caption=True, test_mode=True)  
                caption_used_memory = get_used_memory()
                used_memory = round((caption_used_memory - initial_used_memory) + 0.1,1) 
                log_caption_to_csv(csv_file, caption_model, caption[1], used_memory)
                session_tested_models_count += 1            
            else:
                tested_models_count += 1
            memory_cleanup()
            # Displaying the counts
            print(f"Previously tested models: {tested_models_count}")
            print(f"Tested in current session: {session_tested_models_count}")
            print(f"Models left to test: {total_models_count - tested_models_count - session_tested_models_count}")
            
        if has_win_os():
            subprocess.run(fr'explorer.exe "outputs\tests"', shell=True)
            
        status = "Test process completed!"
    except:
        status="Something went wrong while process caption test."
        print(status)
        traceback.print_exc()
        memory_cleanup()
    return status
         
def test_all_clip_models(image, caption_model_name, load_mode, feature_mode, device):
    try:
        if not image:
            raise Exception("Images must be provided!")
        
        global tested_models_count, session_tested_models_count
        test_path = prep_test_folder()
        total_models_count = len(list_clip_models())
        csv_file = f"{test_path}/model_clip_test_results.csv"
        tested_combinations = load_tested_combinations(csv_file)
        load_config()
        current_caption_model = caption_model_name
        initial_used_memory = get_used_memory()
        for clip_model in sorted(sorted(list_clip_models(), key=lambda t: t[1]), key=lambda t: t[0], reverse=True):        
            if (current_caption_model, clip_model) not in tested_combinations:
                question_prompt_text = get_question_prompt(current_caption_model)
                caption = image_to_prompt(image=image, question_prompt=question_prompt_text, generate_features=True, check_dataset=False, dataset_name='Test', feature_mode=feature_mode, 
                                        precision_type='FP16', load_mode=load_mode, clip_model_name=clip_model, caption_model_name=current_caption_model, device=device,
                                        temperature=0.2, top_p=0.7, max_new_tokens=0, check_txt_caption=False, check_lowvram=False, check_only_caption=True, test_mode=True) 
                clip_used_memory = get_used_memory()
                used_memory = round((clip_used_memory - initial_used_memory - get_caption_model_vram(load_mode, caption_model_name)) + 0.1,1) 
                log_clip_to_csv(csv_file, current_caption_model, clip_model, caption, used_memory)
                session_tested_models_count += 1            
            else:
                tested_models_count += 1
            memory_cleanup()
            # Displaying the counts
            print(f"Previously tested models: {tested_models_count}")
            print(f"Tested in current session: {session_tested_models_count}")
            print(f"Models left to test: {total_models_count - tested_models_count - session_tested_models_count}")
        
        if has_win_os():
            subprocess.run(fr'explorer.exe "outputs\tests"', shell=True)
        
        status = "Test process completed!"
    except:
        status="Something went wrong while process CLIP test."
        print(status)
        traceback.print_exc()
        memory_cleanup()
    return status

def load_config(caption_model_name='blip-large', 
                clip_model_name ='ViT-L-14/openai', 
                device='auto',                
                generate_features=False, 
                precision_type=torch.float16,
                load_4bit=False, 
                load_8bit=False):
    global config
    _device = set_device(device)        
    
    if args.cache_dir is not None:
        config = Config(caption_model_name=caption_model_name, 
                        clip_model_name=clip_model_name,
                        cache_model_path=args.cache_dir, 
                        download_models_to_cache=True,                         
                        generate_features=generate_features, 
                        dtype = precision_type,
                        load_4bit=load_4bit, 
                        load_8bit=load_8bit)
    else:
        config = Config(caption_model_name=caption_model_name, 
                        clip_model_name=clip_model_name,                                                 
                        generate_features=generate_features, 
                        dtype = precision_type,
                        load_4bit=load_4bit,
                        load_8bit=load_8bit)
    config.device=_device    

load_config()

def image_analysis(image, clip_model_name):
            
    validate_reload(None, clip_model_name, check_lowvram=False, device='auto', generate_features=True, precision_type='FP16', load_4bit=False, load_8bit=False)

    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)

    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}
    
    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks

def validate_params(caption_model_name, clip_model_name, device, generate_features, precision_type, load_4bit, load_8bit):
    
    isValid = True 
        
    if generate_features != config.generate_features:
        isValid = False if (generate_features and not config.generate_features) else True
        config.generate_features = generate_features        
    
    if precision_type != config.dtype:
        config.dtype = precision_type
        isValid = False
        
    #if (caption_model_name is not None):    
    if caption_model_name != config.caption_model_name or (not config.caption_model):
        config.caption_model_name = caption_model_name
        isValid = False

    if clip_model_name != config.clip_model_name:
        config.clip_model_name = clip_model_name
        isValid = False
    
    if set_device(device) != config.device:
        config.device = set_device(device)
        isValid = False
             
    if load_4bit != config.load_4bit:
        config.load_4bit = load_4bit
        isValid = False     
    
    if load_8bit != config.load_8bit:
        config.load_8bit = load_8bit
        isValid = False
        
    return isValid
    
def validate_reload(caption_model_name, clip_model_name, check_lowvram, device, generate_features, precision_type, load_4bit, load_8bit):
    
    if not config:
        load_config(caption_model_name, clip_model_name, device, generate_features, precision_type, load_4bit, load_8bit)
    
    isValid = validate_params(caption_model_name, clip_model_name, device, generate_features, precision_type, load_4bit, load_8bit)
    if not isValid:
        if not check_lowvram:
            load_ci() 
        else:            
            config.apply_low_vram_defaults()                          
            load_ci()   

def get_caption_model_vram(toggle_load_mode, caption_model):    
    VRAM = 0    
    if os.path.exists("./clip_interrogator/caption_model_info.json"):
        with open("./clip_interrogator/caption_model_info.json", "r", encoding="utf8") as file:
            caption_model_info = file.read()

        caption_dict = json.loads(caption_model_info)
        data = caption_dict["data"]
        filter_caption_1 = list(filter(lambda x:x["model"]==caption_model, data))
        if(len(filter_caption_1) > 0):
            filter_caption_2 = list(filter(lambda x:x["load_mode"]==toggle_load_mode, filter_caption_1))
        if(len(filter_caption_2) > 0):
            VRAM = filter_caption_2[0]["VRAM"]
    return VRAM

def get_clip_model_vram(clip_model, feature_mode):
    VRAM = 0            
    if clip_model:
        if os.path.exists("./clip_interrogator/clip_model_info.json"):
            with open("./clip_interrogator/clip_model_info.json", "r", encoding="utf8") as file:
                clip_model_info = file.read()

            clip_dict = json.loads(clip_model_info)
            data = clip_dict["data"]
            filter_clip_1 = list(filter(lambda x:x["model"]==clip_model, data))
            if(len(filter_clip_1) > 0):
                filter_clip_2 = list(filter(lambda x:x["feature_mode"]==feature_mode, filter_clip_1))
            if(len(filter_clip_2) > 0):
                VRAM = filter_clip_2[0]["VRAM"]
    return VRAM
    
def get_models_vram(toggle_load_mode, caption_model, clip_model=None, feature_mode=None) :

    VRAM1 = get_caption_model_vram(toggle_load_mode, caption_model)
    VRAM2 = get_clip_model_vram(clip_model, feature_mode)
    
    if VRAM1 or VRAM2:
        totalVRAM = VRAM1+VRAM2
        if totalVRAM >= 40:
            strChar = ">"
        else: 
            strChar = "~"
        return f"<p>🚨<b>{strChar}{str(float('{:.2f}'.format(totalVRAM)))}GB VRAM</b> is required!🚨</p>"
    else:
        return ""

def print_settings(precision_type, 
                   load_mode, 
                   device, 
                   check_lowvram, 
                   generate_features, 
                   check_only_caption,                   
                   check_txt_caption,
                   check_dataset,
                   dataset_name,
                   caption_model,
                   temperature, 
                   top_p, 
                   max_new_tokens, 
                   clip_model,
                   feature_mode,
                   question_prompt
                   ):
    print("Running with this settings...")
    print("-"*100)
    settings =  f"Precision Type: {precision_type}"
    settings += f"\nLoad Mode: {load_mode}" 
    settings += f"\nDevice: {device}"
    settings += f"\nOptimize for Low VRAM: {check_lowvram}" 
    settings += f"\nInclude image features in the prompt: {generate_features}"
    settings += f"\nDon't save dataset images: {check_only_caption}"
    settings += f"\nGenerate individual caption file: {check_txt_caption}"
    settings += f"\nGenerate dataset: {check_dataset}"
    settings += f"\nDataset Name: {dataset_name}"
    settings += f"\nCaption Model: {caption_model}"
    settings += f"\nMax outputs tokens: {max_new_tokens}"
    settings += f"\nTemperature: {temperature}"
    settings += f"\nTop p: {top_p}"
    settings += f"\nCLIP Model: {clip_model}" if generate_features else ""
    settings += f"\nFeature Mode: {feature_mode}" if generate_features else ""
    settings += f"\nQuestion prompt: {question_prompt}" if  caption_model.startswith("llava") or caption_model.startswith('cogvlm') or caption_model.startswith('cogagent') or caption_model.startswith("kosmos-2") else ""
    print(settings)
    print("-"*100)
    
def image_to_prompt(image, 
                    question_prompt, 
                    generate_features, 
                    check_dataset, 
                    dataset_name, 
                    feature_mode, 
                    precision_type,
                    load_mode, 
                    clip_model_name, 
                    caption_model_name, 
                    device,
                    temperature, 
                    top_p, 
                    max_new_tokens, 
                    check_txt_caption, 
                    check_lowvram, 
                    check_only_caption,
                    test_mode=False):
    try:
                
        if not image:
            raise Exception("Images must be provided!")
        
        if dataset_name==None:
            raise Exception("Dataset name must be provided!")
            
        image = image.convert('RGB')
                
        global ci                  
        load_8bit=True if load_mode=='8bit' else False
        load_4bit=True if load_mode=='4bit' else False
                
        if(feature_mode=='negative'):
            caption_model_name = None
        
        _precision_type=torch.float32 if device=='cpu' else (torch.float16 if precision_type=='FP16' else torch.bfloat16)
        max_new_tokens = 2048 if max_new_tokens == 0 else max_new_tokens
        
        print_settings(precision_type, load_mode, device, check_lowvram, generate_features, check_only_caption, check_txt_caption, check_dataset, dataset_name, caption_model_name, temperature, top_p,max_new_tokens, clip_model_name, feature_mode, question_prompt)
        validate_reload(caption_model_name, clip_model_name, check_lowvram, device, generate_features, _precision_type, load_4bit, load_8bit)                                  
        prompt = generate_captions(ci, feature_mode, question_prompt, temperature, top_p, max_new_tokens, check_dataset, dataset_name, [image], check_txt_caption, check_only_caption, test_mode)
        
        if not test_mode and has_win_os():            
            subprocess.run(fr'explorer.exe "outputs\images"', shell=True)            
        
        status ="Image process completed!"         
        print(status)       
        return [status, prompt]       
    except:
        status="Something went wrong while process captions."
        print(status)
        traceback.print_exc()
        memory_cleanup()
        return [status,'']
        
def batch_process(folder, 
                  question_prompt, 
                  generate_features, 
                  check_dataset, 
                  dataset_name, 
                  feature_mode, 
                  precision_type,
                  load_mode, 
                  clip_model_name, 
                  caption_model_name,
                  device, 
                  temperature, 
                  top_p, 
                  max_new_tokens, 
                  check_txt_caption, 
                  check_lowvram, 
                  check_only_caption):
    from PIL import Image
        
    try: 
        if not os.path.isdir(folder):
            raise Exception("A valid image folder must be provided!")
        
        if dataset_name==None:
            raise Exception("Dataset name must be provided!")
         
        print("Batch process started.")
        print("Preparing images from folder...")
        images = []
        global ci
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(folder, filename)
                image = Image.open(img_path).convert('RGB')                
                image.orig_file_name = filename
                images.append(image)

        if len(images) <=0:
            raise Exception("The folder has no images to be processed!")
        
        load_8bit=True if load_mode=='8bit' else False
        load_4bit=True if load_mode=='4bit' else False
        _precision_type=torch.float32 if device=='cpu' else (torch.float16 if precision_type=='FP16' else torch.bfloat16)
        
        print_settings(precision_type, load_mode, device, check_lowvram, generate_features, check_only_caption, check_txt_caption, check_dataset, dataset_name, caption_model_name, temperature, top_p,max_new_tokens, clip_model_name, feature_mode, question_prompt)
        validate_reload(caption_model_name, clip_model_name, check_lowvram, device, generate_features, _precision_type, load_4bit, load_8bit)                                
        generate_captions(ci, feature_mode, question_prompt, temperature, top_p, max_new_tokens, check_dataset, dataset_name, images, check_txt_caption, check_only_caption)
        
        if has_win_os():
            subprocess.run(fr'explorer.exe "outputs\images"', shell=True)
        
        status ="Image process completed!"         
        print(status)       
        return status
    except:
        print("Something went wrong while process captions.")
        traceback.print_exc()
        memory_cleanup()
        pass
    
def get_question_prompt(caption_model_value):
    if caption_model_value.startswith("llava") or caption_model_value.startswith('cogvlm') or caption_model_value.startswith('cogagent'):
        question_prompt_text = "Provide caption for the image in one sentence. Be detailed but precise."
    elif caption_model_value.startswith("kosmos-2"):
        question_prompt_text = "Describe this image in detail:"            
    else:
        question_prompt_text = None

    return question_prompt_text

def prompt_tab():
    str_clip_model_name = config.clip_model_name if config.clip_model_name != None else list_clip_models()[0] 
    str_caption_model_name = config.caption_model_name if config.caption_model_name != None else list_caption_models()[0]
    with gr.Column():
        with gr.Row():
            folder_path = gr.Textbox(label="Folder Path")
            image_folder_input_folder = gr.Button(
                        folder_symbol,
                        elem_id='open_folder_small',
                        visible=True,                        
                    )
            image_folder_input_folder.click(
                open_folder,
                outputs=folder_path,
                show_progress=False,                
            )
        with gr.Row():
            image = gr.Image(type='pil', label="Single image upload (Optional if folder path given)")           
            with gr.Column():               
                with gr.Accordion("Load options", open=True) as load_options_row:
                    with gr.Row():
                        lb_load_mode = gr.HTML(elem_id="loadMode", value="<p>🚨<b>~1,2GB VRAM</b> is required!🚨</p>")
                    with gr.Row():
                        precision_type = gr.Radio(['FP16', 'BF16'], label='Precision type', value='FP16')                         
                        load_mode = gr.Radio(['16bit', '8bit', '4bit'], label='Load mode', min_width=280, value='16bit') 
                        devices = gr.Dropdown(list_devices(), value="auto", label='Device')                                                     
                    check_lowvram = gr.Checkbox(label="Optimize settings for low VRAM (This will always use 'blip-base' model for caption)",value=False)                    
                with gr.Accordion("Generation options", open=True) as generation_options_row:
                    with gr.Row():
                        check_include_features = gr.Checkbox(label="Include image features in the prompt", value=False)            
                    with gr.Row():
                        check_only_caption = gr.Checkbox(label="Don't save dataset images", value=False)
                        check_txt_caption = gr.Checkbox(label="Generate individual caption file", value=True)
                    with gr.Row():
                        check_dataset = gr.Checkbox(label="Generate dataset",value=False)        
                        dataset_name = gr.Textbox(label="Dataset name",value='Default')
        with gr.Tab("Caption") as caption_options_row:
            with gr.Row():
                with gr.Column(min_width=705):                        
                    caption_model = gr.Dropdown(list_caption_models(), value=str_caption_model_name, label='Model')               
                    question_prompt = gr.Textbox(label="Question prompt", value="Provide caption for the image in one sentence. Be detailed but precise.", visible=False)                        
                with gr.Column():                    
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature", visible=False)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", visible=False)
                    max_new_tokens = gr.Slider(minimum=0, maximum=2048, value=2048, step=64, interactive=True, label="Max output tokens")                        
        #with gr.Row(visible=False) as features_options_row:
        with gr.Tab("Features", visible=False) as features_options_row:
            with gr.Row():
                clip_model = gr.Dropdown(list_clip_models(), value=str_clip_model_name, label='CLIP Model',visible=False)
                feature_mode = gr.Radio(['best', 'fast', 'classic', 'negative'], label='Feature Mode', min_width=400, value='classic', visible=False)                                                   

        features_elements = [clip_model, feature_mode, features_options_row]
        def update_features_elements(check_include_features, load_mode_value, caption_model_value, clip_model_value, feature_mode):            
            outputs=[]
            outputs.append(gr.update(visible=check_include_features))    
            outputs.append(gr.update(visible=check_include_features, choices=['best', 'fast', 'classic', 'negative'], value='classic'))    
            outputs.append(gr.update(visible=check_include_features))              
            lb_load_mode = update_load_mode(check_include_features, load_mode_value, caption_model_value, clip_model_value, feature_mode)            
            return lb_load_mode, *outputs       
        
        check_include_features.change(
            fn=update_features_elements,
            inputs=[check_include_features, load_mode, caption_model, clip_model, feature_mode],
            outputs=[lb_load_mode, *features_elements])    
        
        def update_load_mode(check_include_features, toggle_load_mode, caption_model, clip_model, feature_mode):     
            if(check_include_features):                  
                return get_models_vram(toggle_load_mode, caption_model, clip_model, feature_mode)  
            else:
                return get_models_vram(toggle_load_mode, caption_model, None, None)                           
        
        caption_elements=[load_mode, precision_type, top_p, temperature, question_prompt]
        def update_caption_options(check_include_features, load_mode_value, precision_type_value, caption_model_value, clip_model_value, feature_mode):
            lb_load_mode = update_load_mode(check_include_features, load_mode_value, caption_model_value, clip_model_value, feature_mode )
            show_elements = True if caption_model_value.startswith("llava") else False
            show_prompt = True if (caption_model_value.startswith("llava") 
                                   or caption_model_value.startswith("kosmos-2") 
                                   or caption_model_value.startswith('cogvlm') 
                                   or caption_model_value.startswith('cogagent')) else False
            
            question_prompt_text = get_question_prompt(caption_model_value)
                
            outputs=[]
            if caption_model_value.startswith('cogvlm') or caption_model_value.startswith('cogagent'):
                outputs.append(gr.update(visible=True, choices=['16bit', '4bit'], value='4bit'))    
            else:                
                outputs.append(gr.update(visible=True, choices=['16bit', '8bit', '4bit'], value=load_mode_value)) 
                
            if caption_model_value.startswith('git-'): 
                outputs.append(gr.update(visible=True, choices=['FP16'], value='FP16'))
            else:
                outputs.append(gr.update(visible=True, choices=['FP16', 'BF16'], value=precision_type_value)) 
                
            outputs.append(gr.update(visible=show_elements))
            outputs.append(gr.update(visible=show_elements))
            outputs.append(gr.update(visible=show_prompt, value=question_prompt_text))                
            
            return lb_load_mode, *outputs
                
        load_mode.change(
            fn=update_load_mode,
            inputs=[check_include_features, load_mode, caption_model, clip_model, feature_mode],
            outputs=[lb_load_mode]
        )
        
        caption_model.change(
            fn=update_caption_options,
            inputs=[check_include_features, load_mode, precision_type, caption_model, clip_model, feature_mode],
            outputs=[lb_load_mode, *caption_elements]
        )
        
        clip_model.change(
            fn=update_load_mode,
            inputs=[check_include_features, load_mode, caption_model, clip_model, feature_mode],
            outputs=[lb_load_mode]
        )
        feature_mode.change(
            fn=update_load_mode,
            inputs=[check_include_features, load_mode, caption_model, clip_model, feature_mode],
            outputs=[lb_load_mode]
        )
        prompt = gr.Textbox(label="Generated prompt for single image")
        status = gr.Textbox(label="Processing status")
        generate_button = gr.Button("Generate prompt for single image")
        batch_button = gr.Button("Batch process for folder")        
        generate_button.click(image_to_prompt, inputs=[image, question_prompt, check_include_features, check_dataset, dataset_name, feature_mode, precision_type, load_mode, clip_model, caption_model, devices, temperature, top_p, max_new_tokens, check_txt_caption, check_lowvram, check_only_caption], outputs=[status,prompt])        
        batch_button.click(batch_process, inputs=[folder_path, question_prompt, check_include_features, check_dataset, dataset_name, feature_mode, precision_type, load_mode, clip_model, caption_model,devices, temperature, top_p, max_new_tokens, check_txt_caption, check_lowvram, check_only_caption], outputs=status)
        with gr.Row():
            test_all_caption_models_button = gr.Button("Test all caption models")
            test_all_caption_models_button.click(test_all_caption_models, inputs=[image, load_mode, devices], outputs=status)      
            test_all_clip_models_button = gr.Button("Test all CLIP models")
            test_all_clip_models_button.click(test_all_clip_models, inputs=[image, caption_model, load_mode, feature_mode, devices], outputs=status)      
          
def analyze_tab():
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            model = gr.Dropdown(list_clip_models(), value='ViT-L-14/openai', label='CLIP Model')
        with gr.Row():
            medium = gr.Label(label="Medium", num_top_classes=5)
            artist = gr.Label(label="Artist", num_top_classes=5)        
            movement = gr.Label(label="Movement", num_top_classes=5)
            trending = gr.Label(label="Trending", num_top_classes=5)
            flavor = gr.Label(label="Flavor", num_top_classes=5)
        analyze_button = gr.Button("Analyze")
        analyze_button.click(image_analysis, inputs=[image, model], outputs=[medium, artist, movement, trending, flavor])


with gr.Blocks(css=css,title=f"IMAGE Interrogator GUI {release}",theme=gr.themes.Default()) as ui:
    gr.HTML("<h1><center>🕵️‍♂️ IMAGE Interrogator 🕵️‍♂️</center></h1>")
    with gr.Tab("Prompt"):
        prompt_tab()
    with gr.Tab("Analyze"):
        analyze_tab()


ui.queue().launch(debug=True,  inline=False,share=args.share)

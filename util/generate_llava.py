"""
Copied and adapted from https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py and https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/model_worker.py
"""
import json
from io import BytesIO
import os, shutil
import datetime
from PIL import Image
import requests
import torch
from util.generate_dataset import save_dataset
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


LLAVA_MODELS = {
    'llava-v1.5-7b': 'liuhaotian/llava-v1.5-7b',
    'llava-v1.5-13b': 'liuhaotian/llava-v1.5-13b'    
}

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if not isinstance(image_file, Image.Image):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
    else:
        return image_file.convert("RGB")


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_llava_model(args, tokenizer, model, image_processor, context_len, model_name):
    # Model
    disable_torch_init()

    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if not isinstance(args.image_file, list):
        images = image_parser(args)
    else:
        images = load_images(args.image_file)
        
    images_tensor = process_images(images, image_processor, model.config)
    if type(images_tensor) is list:
        images_tensor = [image.to(model.device, dtype=torch.float16) for image in images_tensor]
    else:
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)
    
    image_sizes = [image.size for image in images] 

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    #stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
   
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor, 
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,                                                                        
            use_cache=True,)
            #stopping_criteria=[stopping_criteria])
  
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return outputs


def generate_captions(model_name, prompt, check_dataset, dataset_name, images, load_8bit, load_4bit, device, temperature, top_p, max_new_tokens, check_txt_caption, check_only_caption, cache_dir=None):
    #its only works for LLaVA 1.5
    model_path = LLAVA_MODELS[model_name]
    
    if cache_dir is not None:
        cached_model_path = os.path.join(cache_dir, model_path) 
    
    if cache_dir is not None and not os.path.exists(cached_model_path):
        from huggingface_hub import snapshot_download
        try:                    
            snapshot_download(repo_id=model_path, local_dir=cached_model_path, local_dir_use_symlinks=False)                
        except Exception as e:
            print(f"Failed to download model: {model_path} from HF")
            print(e)
            return False

    model_path = cached_model_path if cache_dir is not None and os.path.exists(cached_model_path) else model_name
         
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, model_base=None, model_name=model_name, load_8bit=load_8bit, load_4bit=load_4bit, device=device, local_files_only=(cache_dir is not None)
    )

    general_prompt = (
        prompt
    )
    
    batch_size = 8
    batch_count = 0
    data_dict = {}
    
    t = datetime.datetime.now()
    image_path = os.path.join(os.getcwd(),'outputs','images',f"{t.year}-{t.month:02d}-{t.day:02d}", dataset_name)
    dataset_path = os.path.join(image_path,'dataset')
    
    if os.path.exists(image_path):
        shutil.rmtree(image_path)
    os.makedirs(image_path, exist_ok=True)
        
    if check_dataset==True:
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.makedirs(dataset_path, exist_ok=True)
  
    for i in range(0, len(images), batch_size):
        batch_count += 1
        images_ = images[i : i + batch_size]
        args = type(
            "Args",
            (),
            {
                "model_path": model_path,
                "model_base": None,
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "query": general_prompt,
                "conv_mode": None,
                "image_file": images_,
                "sep": ",",
            },
        )()
        print(f"Generating caption for images... Batch: {batch_count} | Total Images: {len(images_)}")
        generated_captions = eval_llava_model(
            args, tokenizer, model, image_processor, context_len, model_name
        )

        for j, image in enumerate(images_):            
            filename = f"{images_[j].orig_file_name}"
            file_path = os.path.join(image_path, filename)            
            if check_only_caption == False:
                print(f"Saving file...: {file_path}")
                image.save(file_path)
            if check_txt_caption == True:
                caption_file = os.path.join(image_path, os.path.splitext(filename)[0] + '.txt')
                print(f"Generating individual file caption...: {caption_file}") 
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(generated_captions[j])
                    
            data_dict.update({filename: generated_captions[j]})

    print(f"Saving dataset metadata...")
    with open(os.path.join(os.getcwd(),'outputs','images',f"{t.year}-{t.month:02d}-{t.day:02d}", dataset_name, f"metadata.jsonl"), "w") as f:
        json.dump(data_dict, f, indent=4)
    
    if check_dataset==True and check_only_caption==False:
        save_dataset(image_path, dataset_path)
        
    if len(images) == 1:
        return generated_captions[0]



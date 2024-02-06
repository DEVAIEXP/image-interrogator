
import json
import os, shutil
import datetime
from util.generate_dataset import save_dataset
from util.generate_llava import eval_llava_model

    
def generate_captions(ci, mode, question_prompt, temperature, top_p, caption_max_length, check_dataset, dataset_name, images, check_txt_caption, check_only_caption, test_mode=False):
        
    batch_size = 8
    batch_count = 0
    data_dict = {}
    isLlavaModel = ci.config.caption_model_name and ci.config.caption_model_name.startswith('llava')
    
    t = datetime.datetime.now()
    image_path = os.path.join(os.getcwd(),'outputs','images',f"{t.year}-{t.month:02d}-{t.day:02d}", dataset_name)
    dataset_path = os.path.join(image_path,'dataset')
    
    if not test_mode:
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
        print(f"Generating caption for images... Batch: {batch_count} | Total Images: {len(images_)}")
        
        generated_captions = [] 
        
        if isLlavaModel:            
            for j, image in enumerate(images_):                        
                args = type(
                    "Args",
                    (),
                    {
                        "model_path": ci.config.model_path,
                        "model_base": None,
                        "model_name": ci.config.caption_model_name,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": caption_max_length,
                        "query": question_prompt,
                        "conv_mode": None,
                        "image_file": [image],
                        "sep": ",",
                    },
                )()       
            
                caption = eval_llava_model(
                    args, ci.tokenizer, ci.caption_model, ci.caption_processor, ci.context_len, ci.config.caption_model_name
                )
                if mode is None or mode =='fast':
                    generated_captions.append(ci.interrogate_fast(image, question_prompt=question_prompt, caption=caption))
                elif mode == 'best':
                    generated_captions.append(ci.interrogate(image, question_prompt=question_prompt, caption=caption))
                elif mode == 'classic':
                    generated_captions.append(ci.interrogate_classic(image, question_prompt=question_prompt, caption=caption))
              
        else:            
            for j, image in enumerate(images_):                        
                if mode is None or mode =='fast':
                    generated_captions.append(ci.interrogate_fast(image, caption_max_length = caption_max_length, question_prompt=question_prompt))
                elif mode == 'best':
                    generated_captions.append(ci.interrogate(image, caption_max_length = caption_max_length, question_prompt=question_prompt))
                elif mode == 'classic':
                    generated_captions.append(ci.interrogate_classic(image, caption_max_length = caption_max_length, question_prompt=question_prompt))                
                elif mode == 'negative':
                    generated_captions.append(ci.interrogate_negative(image))
         
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
    
    if not test_mode:
        print(f"Saving dataset metadata...")
        with open(os.path.join(os.getcwd(),'outputs','images',f"{t.year}-{t.month:02d}-{t.day:02d}", dataset_name, f"metadata.jsonl"), "w") as f:
            json.dump(data_dict, f, indent=4)
        
        if check_dataset==True and check_only_caption==False:
            save_dataset(image_path, dataset_path)
        
    if len(images) == 1:       
        return data_dict[list(data_dict)[0]]    
    

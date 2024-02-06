import json
import os

VRAM1 = 0
VRAM2 = 0

if os.path.exists("./clip_interrogator/caption_model_info.json"):
        with open("./clip_interrogator/caption_model_info.json", "r", encoding="utf8") as file:
            caption_model_info = file.read()

        caption_dict = json.loads(caption_model_info)
        data = caption_dict["data"]
        filter_caption_1 = list(filter(lambda x:x["model"]=="blip-base", data))
        filter_caption_2 = list(filter(lambda x:x["load_mode"]=="8bit", filter_caption_1))
        VRAM1 = filter_caption_1[0]["VRAM"]

if os.path.exists("./clip_interrogator/clip_model_info.json"):
        with open("./clip_interrogator/clip_model_info.json", "r", encoding="utf8") as file:
            clip_model_info = file.read()

        clip_dict = json.loads(clip_model_info)
        data = clip_dict["data"]
        filter_clip_1 = list(filter(lambda x:x["model"]=="RN50/openai", data))
        filter_clip_2 = list(filter(lambda x:x["feature_mode"]=="fast", filter_clip_1))
        VRAM2 = filter_clip_2[0]["VRAM"]

if VRAM1 or VRAM2:
        print(float("{:.2f}".format(VRAM1 + VRAM2)))
else:
        print("")

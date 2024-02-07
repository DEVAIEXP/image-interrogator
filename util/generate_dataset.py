from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import json

def save_dataset(image_path, dataset_path):
    with open(f"{image_path}/metadata.jsonl") as f:
        data_dict = json.load(f)

    def generation_fn():
        for image, text in data_dict.items():
            yield {
                "image": {"path": f"{image_path}/{image}"},
                "text": text
            }

    ds = Dataset.from_generator(
        generation_fn,
        features=Features(
            image=ImageFeature(),
            text=Value("string")
        ),
    )    
    ds.save_to_disk(f"{dataset_path}")

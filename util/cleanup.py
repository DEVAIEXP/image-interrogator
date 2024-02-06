import torch
import gc

def memory_cleanup():
    gc.collect()  # Explicit garbage collection
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release GPU memory
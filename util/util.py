import torch
import gc
import platform
import sys

def memory_cleanup():
    gc.collect()  # Explicit garbage collection
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release GPU memory
        
def get_used_memory():
    v = torch.cuda.mem_get_info()
    reserved = v[0] / (1024**3) 
    total = v[1] / (1024**3) 
    used = round(total - reserved,1)
    return used

def get_reserved_memory():
    v = torch.cuda.mem_get_info()
    reserved = v[0] / (1024**3)     
    return reserved

def get_total_memory():
    v = torch.cuda.mem_get_info()
    total = v[1] / (1024**3) 
    return total

def has_win_os():    
    platform_name = platform.uname()
    if sys.platform == 'win32' or 'WSL2' in platform_name.release:
        return True
    return False
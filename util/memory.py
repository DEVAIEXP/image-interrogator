import torch

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
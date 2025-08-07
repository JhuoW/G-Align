import random
import os
import numpy as np
import torch
import os.path as osp

def seed_setting(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)

def pth_safe_save(obj, path):
    if obj is not None:
        torch.save(obj, path)
        
def pth_safe_load(path):
    if osp.exists(path):
        return torch.load(path, weights_only=False)
    return None
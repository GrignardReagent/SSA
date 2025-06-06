import torch
import numpy as np
import random

def set_seed(seed=42):
    """
    Set random seed for reproducibility across torch, numpy, and random.

    Args:
        seed (int): Random seed value (default: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_seed(seed=42):
    """
    Set seed for reproducibility across multiple libraries.
    
    Args:
        seed (int): Seed value to use. Default is 42.
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For some operations in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
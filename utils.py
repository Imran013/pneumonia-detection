import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from datetime import datetime

class Utils:

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f'Random seed set to {seed}')

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str,int]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total' : total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    @staticmethod
    def format_time(seconds : float) -> str:
        hours = int(seconds//3600)
        minutes = int((seconds%3600)//60)
        secs = int(seconds%60)

        if hours > 0:
            return f'{hours}h {minutes}m {secs}s'
        elif minutes > 0:
            return f'{minutes}m {secs}s'
        else:
            return f'{secs}s'
        
    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S')

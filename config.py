from dataclasses import dataclass, field
from typing import Tuple, Optional
import os

@dataclass
class Config:
    #directory
    data_root: str = 'chest_xray'

    #Model architecture
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 2
    in_channels: int = 3

    #Hyperparameters
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3

    #Optimizer
    optimizer_type: str = 'adam'
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9

    #Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'step'
    step_size: int = 7
    gamma: float = 0.1

    #Regularization
    dropout_rate: float = 0.5
    use_batch_norm: bool = True

    #Data loading
    num_workers: int = 0
    pin_memory: bool = False

    #Data Augmentation (standard - both classes)
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.6
    rotation_degrees: float = 15
    color_jitter: Tuple[float, float] = (0.2, 0.2)

    #Data Augmentation (aggressive - NORMAL class only)
    normal_rotation_degrees: float = 30
    normal_color_jitter: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.1)
    normal_random_affine_degrees: float = 20
    normal_random_affine_translate: Tuple[float, float] = (0.1, 0.1)
    normal_random_affine_scale: Tuple[float, float] = (0.85, 1.15)
    normal_perspective_prob: float = 0.3
    normal_gaussian_blur_kernel: int = 3
    normal_random_erasing_prob: float = 0.2

    # Normalization (ImageNet statistics)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    #Training strategy
    gradient_accumulation_steps: int = 2
    early_stopping_patience: int = 20
    save_best_only: bool = True
    label_smoothing: float = 0.1

    #Balanced sampling
    use_weighted_sampler: bool = True

    #Device
    device: str = 'cuda'

    #Logging
    log_interval: int = 10

    #Random seed
    seed: int = 42

    #Auto-generated paths
    train_dir: str = field(default = '', init = False)
    val_dir: str = field(default = '', init = False)
    test_dir: str = field(default = '', init = False)
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    plot_dir: str = 'plots'

    def __post_init__(self):

        self.train_dir = os.path.join(self.data_root,'train')
        self.val_dir = os.path.join(self.data_root, 'val')
        self.test_dir = os.path.join(self.data_root, 'test')

        for directory in [self.checkpoint_dir, self.log_dir, self.plot_dir]:
            os.makedirs(directory, exist_ok = True)

    def validate(self) -> None:
        if not os.path.exists(self.data_root):
            raise ValueError(f'Data root not found {self.data_root}')
        
        for path in [self.train_dir, self.val_dir, self.test_dir]:
            if not os.path.exists(path):
                raise ValueError(f'Directory not found {path}')
            
        print('Configuration validated')
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
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    #Optimizer
    optimizer_type: str = 'adam'
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9

    #Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'
    step_size: int = 7
    gamma: float = 0.1
    warmup_epochs: int = 3
    min_lr: float = 1e-6


    #Regularization
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    use_mixup: bool = True
    use_cutmix: bool = True


    #Data loading
    num_workers: int = 4
    pin_memory: bool = True

    #Data Augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 20
    color_jitter: Tuple[float, float, float, float] = (0.2, 0.2, 0.1, 0.1)
    random_affine: bool = True
    random_perspective: bool = True
    random_erasing_prob: float = 0.2
    gaussian_blur_prob: float = 0.1



    # Normalization (ImageNet statistics)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    #Training strategy
    gradient_accumlation_steps: int = 1
    early_stopping_patience: int = 10
    save_best_only: bool = True

    #Gradient clipping
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0

    #EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    #Device
    device: str = 'cpu'

    #Logging
    log_interval: int = 10
    save_interval: int = 5

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
            
        print('Configration validated')





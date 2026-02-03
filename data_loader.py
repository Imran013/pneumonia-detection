import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np

class Config:
    """Placeholder - actual config import olunmalıdır"""
    pass

class MixUp:
    """MixUp data augmentation"""
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam

class CutMix:
    """CutMix data augmentation"""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size)
        
        # Get random box
        _, _, h, w = images.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam

class DataManager:
    """Enhanced DataManager with advanced augmentation"""
    
    def __init__(self, config):
        self.config = config
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _create_train_transform(self) -> transforms.Compose:
        """Create training transforms with STRONG augmentation"""
        transform_list = [transforms.Resize(self.config.image_size)]

        if self.config.use_augmentation:
            # Basic augmentations
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
                transforms.RandomRotation(degrees=self.config.rotation_degrees),
            ])
            
            # Vertical flip (medical images üçün faydalı)
            if hasattr(self.config, 'vertical_flip_prob'):
                transform_list.append(
                    transforms.RandomVerticalFlip(p=self.config.vertical_flip_prob)
                )
            
            # Color jitter - improved
            if len(self.config.color_jitter) == 4:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=self.config.color_jitter[0],
                        contrast=self.config.color_jitter[1],
                        saturation=self.config.color_jitter[2],
                        hue=self.config.color_jitter[3]
                    )
                )
            else:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=self.config.color_jitter[0],
                        contrast=self.config.color_jitter[1]
                    )
                )
            
            # Advanced augmentations
            if hasattr(self.config, 'random_affine') and self.config.random_affine:
                transform_list.append(
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        shear=5
                    )
                )
            
            if hasattr(self.config, 'random_perspective') and self.config.random_perspective:
                transform_list.append(
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.3)
                )
            
            # Gaussian blur
            if hasattr(self.config, 'gaussian_blur_prob') and self.config.gaussian_blur_prob > 0:
                transform_list.append(
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                    ], p=self.config.gaussian_blur_prob)
                )

        # Always apply these
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
        
        # Random erasing (after normalization)
        if hasattr(self.config, 'random_erasing_prob') and self.config.random_erasing_prob > 0:
            transform_list.append(
                transforms.RandomErasing(
                    p=self.config.random_erasing_prob,
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3)
                )
            )

        return transforms.Compose(transform_list)

    def _create_test_transform(self) -> transforms.Compose:
        """Create test transforms (no augmentation)"""
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def load_datasets(self) -> None:
        """Load all datasets"""
        print('📂 Loading datasets...')

        train_transform = self._create_train_transform()
        test_transform = self._create_test_transform()

        try:
            self.train_dataset = datasets.ImageFolder(
                root=self.config.train_dir,
                transform=train_transform
            )

            self.val_dataset = datasets.ImageFolder(
                root=self.config.val_dir,
                transform=test_transform
            )

            self.test_dataset = datasets.ImageFolder(
                root=self.config.test_dir,
                transform=test_transform
            )

            self._print_dataset_stats()

        except FileNotFoundError as e:
            raise FileNotFoundError(f'Dataset directory not found: {e}')
        
    def _print_dataset_stats(self) -> None:
        """Print dataset statistics"""
        print('\n📊 Dataset Statistics:')
        print(f"  Training samples:   {len(self.train_dataset):>6,}")
        print(f"  Validation samples: {len(self.val_dataset):>6,}")
        print(f"  Test samples:       {len(self.test_dataset):>6,}")
        print(f"  Total samples:      {len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset):>6,}")
        print(f"\n  Classes: {self.train_dataset.classes}")
        print(f"  Class indices: {self.train_dataset.class_to_idx}")
        
        # Class balance check
        train_targets = [label for _, label in self.train_dataset.samples]
        class_counts = np.bincount(train_targets)
        print(f"\n  Class distribution (train):")
        for i, (cls_name, count) in enumerate(zip(self.train_dataset.classes, class_counts)):
            print(f"    {cls_name}: {count} ({count/len(train_targets)*100:.1f}%)")

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloaders with improved settings"""
        print('\n🔄 Creating dataloaders...')

        if self.train_dataset is None:
            raise RuntimeError('Datasets not loaded. Call load_datasets() first.')
        
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True  # ✨ Yeni: son batch-ı at (batch norm üçün)
        )

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        print(f'✅ Dataloaders created:')
        print(f'  Batch size: {self.config.batch_size}')
        print(f'  Train batches: {len(train_loader)}')
        print(f'  Val batches: {len(val_loader)}')
        print(f'  Test batches: {len(test_loader)}')

        return train_loader, val_loader, test_loader
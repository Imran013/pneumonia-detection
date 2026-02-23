import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from typing import Tuple, Optional
from config import Config


class ClassAwareDataset(Dataset):
    """NORMAL class-a gucluu augmentation, PNEUMONIA-ya standart augmentation tetbiq edir."""

    def __init__(self, root: str, normal_transform: transforms.Compose, pneumonia_transform: transforms.Compose):
        self.dataset = datasets.ImageFolder(root=root)
        self.normal_transform = normal_transform
        self.pneumonia_transform = pneumonia_transform

        # ImageFolder attributes
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.targets = self.dataset.targets
        self.samples = self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = self.dataset.loader(path)

        # NORMAL = 0, PNEUMONIA = 1
        if label == 0:
            image = self.normal_transform(image)
        else:
            image = self.pneumonia_transform(image)

        return image, label


class DataManager:

    def __init__(self, config: Config):
        self.config = config
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _create_normal_transform(self) -> transforms.Compose:
        """NORMAL class ucun guclu augmentation"""
        transform_list = [transforms.Resize(self.config.image_size)]

        if self.config.use_augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
                transforms.RandomRotation(degrees=self.config.normal_rotation_degrees),
                transforms.RandomAffine(
                    degrees=self.config.normal_random_affine_degrees,
                    translate=self.config.normal_random_affine_translate,
                    scale=self.config.normal_random_affine_scale
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.2,
                    p=self.config.normal_perspective_prob
                ),
                transforms.ColorJitter(
                    brightness=self.config.normal_color_jitter[0],
                    contrast=self.config.normal_color_jitter[1],
                    saturation=self.config.normal_color_jitter[2],
                    hue=self.config.normal_color_jitter[3]
                ),
                transforms.GaussianBlur(
                    kernel_size=self.config.normal_gaussian_blur_kernel,
                    sigma=(0.1, 2.0)
                ),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std),
        ])

        if self.config.use_augmentation:
            transform_list.append(
                transforms.RandomErasing(p=self.config.normal_random_erasing_prob)
            )

        return transforms.Compose(transform_list)

    def _create_pneumonia_transform(self) -> transforms.Compose:
        """PNEUMONIA class ucun standart augmentation"""
        transform_list = [transforms.Resize(self.config.image_size)]

        if self.config.use_augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
                transforms.RandomRotation(degrees=self.config.rotation_degrees),
                transforms.ColorJitter(
                    brightness=self.config.color_jitter[0],
                    contrast=self.config.color_jitter[1]
                )
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std)
        ])

        return transforms.Compose(transform_list)

    def _create_test_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std)
        ])

    def load_datasets(self) -> None:
        print('Loading datasets...')

        normal_transform = self._create_normal_transform()
        pneumonia_transform = self._create_pneumonia_transform()
        test_transform = self._create_test_transform()

        try:
            # Train: class-aware augmentation
            self.train_dataset = ClassAwareDataset(
                root=self.config.train_dir,
                normal_transform=normal_transform,
                pneumonia_transform=pneumonia_transform
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
            raise FileNotFoundError(f'Dataset directory not found {e}')

    def _print_dataset_stats(self) -> None:
        print('\nDataset stats')
        print(f"  Training samples:   {len(self.train_dataset):>6,}")
        print(f"  Validation samples: {len(self.val_dataset):>6,}")
        print(f"  Test samples:       {len(self.test_dataset):>6,}")
        print(f"  Total samples:      {len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset):>6,}")
        print(f"\n  Classes: {self.train_dataset.classes}")
        print(f"  Class indices: {self.train_dataset.class_to_idx}")

        # Class distribution
        targets = self.train_dataset.targets
        class_counts = torch.bincount(torch.tensor(targets))
        print(f"\n  Class distribution (train):")
        for i, cls_name in enumerate(self.train_dataset.classes):
            print(f"    {cls_name}: {class_counts[i]:,} samples ({100*class_counts[i]/len(targets):.1f}%)")

    def _create_weighted_sampler(self) -> WeightedRandomSampler:
        """Her epoch-da NORMAL ve PNEUMONIA beraber sayda secilsin"""
        targets = self.train_dataset.targets
        class_counts = torch.bincount(torch.tensor(targets))

        # Her class ucun weight: 1 / class_count
        class_weights = 1.0 / class_counts.float()

        # Her sample-a oz class-inin weight-ini ver
        sample_weights = torch.tensor([class_weights[t] for t in targets])

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(targets),
            replacement=True
        )

        print(f"\n  WeightedRandomSampler aktiv:")
        print(f"    NORMAL weight:    {class_weights[0]:.6f}")
        print(f"    PNEUMONIA weight: {class_weights[1]:.6f}")
        print(f"    Ratio:            {class_weights[0]/class_weights[1]:.2f}x")

        return sampler

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        print('Creating dataloaders')

        if self.train_dataset is None:
            raise RuntimeError('Dataset not loaded. Call load_datasets() first.')

        # WeightedRandomSampler istifade olunursa shuffle=False olmalidir
        if self.config.use_weighted_sampler:
            sampler = self._create_weighted_sampler()
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        else:
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
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

        print(f'\nDataloaders created.')
        print(f'  Batch size: {self.config.batch_size}')
        print(f'  Train batches: {len(train_loader)}')
        print(f'  Val batches:   {len(val_loader)}')
        print(f'  Test batches:  {len(test_loader)}')
        print(f'  Weighted sampler: {self.config.use_weighted_sampler}')

        return train_loader, val_loader, test_loader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List
import time
import os

from config import Config
from utils import Utils

class Trainer:

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            config: Config
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.model.to(config.device)

        # ═══════════════════════════════════════════════════
        # CLASS WEIGHTS - imbalance ucun
        # ═══════════════════════════════════════════════════
        class_weights = self._calculate_class_weights()
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            weight=class_weights
        )

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        print("\n" + "="*70)
        print("TRAINER INITIALIZED")
        print("="*70)
        print(f"Optimizer: {config.optimizer_type.upper()}")
        print(f"Learning Rate: {config.learning_rate}")
        print(f"Device: {config.device.upper()}")
        print(f"Label Smoothing: {config.label_smoothing}")
        print(f"Class Weights: ACTIVE")

    def _calculate_class_weights(self) -> torch.Tensor:
        """Training dataset-den class weight hesabla"""
        try:
            dataset = self.train_loader.dataset
            if hasattr(dataset, 'targets'):
                targets = dataset.targets
            elif hasattr(dataset, 'samples'):
                targets = [s[1] for s in dataset.samples]
            else:
                targets = [0] * 1140 + [1] * 3294

            class_counts = torch.bincount(torch.tensor(targets)).float()
        except Exception:
            print("  Class distribution dataset-den alinmadi, default istifade olunur")
            class_counts = torch.tensor([1140.0, 3294.0])

        total = class_counts.sum()
        num_classes = len(class_counts)
        class_weights = total / (num_classes * class_counts)
        class_weights = class_weights.to(self.config.device)

        print("\n" + "="*70)
        print("CLASS WEIGHTS (Balanced Training)")
        print("="*70)
        print(f"  NORMAL:    {int(class_counts[0]):,} samples -> weight {class_weights[0]:.4f}")
        print(f"  PNEUMONIA: {int(class_counts[1]):,} samples -> weight {class_weights[1]:.4f}")
        print(f"  NORMAL {class_weights[0]/class_weights[1]:.2f}x daha cox oneme malik olacaq")

        return class_weights

    def _create_optimizer(self) -> optim.Optimizer:
        if self.config.optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f'Unknown optimizer {self.config.optimizer_type}')
        
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        if self.config.scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        else:
            raise ValueError(f'Unknown scheduler {self.config.scheduler_type}')
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):

            inputs = inputs.to(self.config.device)
            labels = labels.to(self.config.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                print(f'  Batch [{batch_idx+1}/{len(self.train_loader)}] '
                      f'Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')
        
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'time': epoch_time
        }

    def validate(self) -> Dict[str, float]:

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return {
            'loss': val_loss,
            'accuracy': val_acc
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved {filepath}')

    def train(self) -> Dict[str, List[float]]:

        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)

        total_start_time = time.time()

        for epoch in range(self.config.num_epochs):

            self.current_epoch = epoch + 1

            print(f"\n{'='*70}")
            print(f"EPOCH {self.current_epoch}/{self.config.num_epochs}")
            print(f"{'='*70}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            print(f"\nEpoch {self.current_epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
            print(f"  LR: {current_lr:.6f} | Time: {Utils.format_time(train_metrics['time'])}")           

            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0

                if self.config.save_best_only:
                    checkpoint_path = os.path.join(
                        self.config.checkpoint_dir,
                        f'best_model_acc_{self.best_val_acc:.2f}.pth'
                    )
                    self.save_checkpoint(checkpoint_path)
                print(f'New best validation acc {self.best_val_acc:.2f}%')
            else:
                self.epochs_without_improvement += 1

            if self.scheduler:
                self.scheduler.step()

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f'\nEarly stop triggered after {self.current_epoch} epochs')
                print(f'No improvement for {self.config.early_stopping_patience} epochs')
                break

        total_time = time.time() - total_start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Total training time: {Utils.format_time(total_time)}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.history
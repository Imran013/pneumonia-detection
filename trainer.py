import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import time
import os
import numpy as np

class Trainer:
    """Enhanced Trainer with class weights for balanced training"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        model_ema=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_ema = model_ema

        self.model.to(config.device)

        
        
        # Calculate class weights from training dataset
        # Get actual class distribution from dataloader
        try:
            # Try to get from dataset
            if hasattr(train_loader.dataset, 'targets'):
                targets = train_loader.dataset.targets
            elif hasattr(train_loader.dataset, 'samples'):
                targets = [s[1] for s in train_loader.dataset.samples]
            else:
                # Default values if can't get from dataset
                # NORMAL: 1140, PNEUMONIA: 3294 (from your dataset)
                targets = [0]*1140 + [1]*3294
            
            # Count classes
            class_counts = torch.bincount(torch.tensor(targets))
            class_counts = class_counts.float()
            
        except Exception as e:
            print(f"⚠️ Could not get class distribution from dataset, using defaults")
            class_counts = torch.tensor([1140.0, 3294.0])
        
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        
        # Inverse frequency weighting
        # weight_i = total_samples / (num_classes * count_i)
        class_weights = total_samples / (num_classes * class_counts)
        class_weights = class_weights.to(config.device)
        
        print("\n" + "="*70)
        print("⚖️  CLASS WEIGHTS (Balanced Training)")
        print("="*70)
        print(f"  Class distribution:")
        print(f"    NORMAL:     {int(class_counts[0]):,} samples")
        print(f"    PNEUMONIA:  {int(class_counts[1]):,} samples")
        print(f"\n  Calculated weights:")
        print(f"    NORMAL:     {class_weights[0]:.4f}")
        print(f"    PNEUMONIA:  {class_weights[1]:.4f}")
        print(f"    Ratio:      {class_weights[0]/class_weights[1]:.2f}x")
        print(f"\n  💡 NORMAL class will receive {class_weights[0]/class_weights[1]:.2f}x more importance in loss")
        
        # ═══════════════════════════════════════════════════
        # LOSS FUNCTION with Class Weights
        # ═══════════════════════════════════════════════════
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=getattr(config, 'label_smoothing', 0.1),
            weight=class_weights  # ✅ CLASS WEIGHTS APPLIED
        )

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.global_step = 0

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # MixUp/CutMix
        self.use_mixup = getattr(config, 'use_mixup', False)
        self.use_cutmix = getattr(config, 'use_cutmix', False)
        self.mixup_alpha = getattr(config, 'mixup_alpha', 0.2)
        self.cutmix_alpha = getattr(config, 'cutmix_alpha', 1.0)

        print("\n" + "="*70)
        print("TRAINER INITIALIZED")
        print("="*70)
        print(f"Optimizer: {config.optimizer_type.upper()}")
        print(f"Learning Rate: {config.learning_rate:.2e}")
        print(f"Device: {config.device.upper()}")
        print(f"Gradient Clipping: {getattr(config, 'use_gradient_clipping', False)}")
        print(f"Model EMA: {model_ema is not None}")
        print(f"MixUp: {self.use_mixup} (alpha={self.mixup_alpha})")
        print(f"CutMix: {self.use_cutmix} (alpha={self.cutmix_alpha})")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.config.optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'adamw':
            return optim.AdamW(
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
            raise ValueError(f'Unknown optimizer: {self.config.optimizer_type}')
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if not self.config.use_scheduler:
            return None
        
        if self.config.scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=getattr(self.config, 'min_lr', 1e-6)
            )
        else:
            raise ValueError(f'Unknown scheduler: {self.config.scheduler_type}')

    def _mixup_data(self, x, y):
        """Apply MixUp augmentation"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _cutmix_data(self, x, y):
        """Apply CutMix augmentation"""
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # Get random box
        _, _, h, w = x.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    def _mixup_criterion(self, pred, y_a, y_b, lam):
        """Loss for MixUp/CutMix"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.config.device)
            labels = labels.to(self.config.device)

            # Apply MixUp or CutMix randomly
            use_mixup = False
            if self.use_mixup and self.use_cutmix:
                if np.random.rand() < 0.5:
                    inputs, labels_a, labels_b, lam = self._mixup_data(inputs, labels)
                    use_mixup = True
                else:
                    inputs, labels_a, labels_b, lam = self._cutmix_data(inputs, labels)
                    use_mixup = True
            elif self.use_mixup:
                inputs, labels_a, labels_b, lam = self._mixup_data(inputs, labels)
                use_mixup = True
            elif self.use_cutmix:
                inputs, labels_a, labels_b, lam = self._cutmix_data(inputs, labels)
                use_mixup = True

            # Forward pass
            outputs = self.model(inputs)

            # Calculate loss
            if use_mixup:
                loss = self._mixup_criterion(outputs, labels_a, labels_b, lam)
            else:
                loss = self.criterion(outputs, labels)

            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if getattr(self.config, 'use_gradient_clipping', False):
                    max_norm = getattr(self.config, 'max_grad_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.model_ema is not None:
                    self.model_ema.update()
                
                self.global_step += 1

            # Statistics
            running_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            
            if use_mixup:
                # Approximate accuracy for mixup
                correct += (lam * predicted.eq(labels_a).sum().item() + 
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                correct += predicted.eq(labels).sum().item()

            # Logging
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

    def validate(self, use_ema: bool = False) -> Dict[str, float]:
        """Validate model"""
        if use_ema and self.model_ema is not None:
            self.model_ema.apply_shadow()
        
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
        
        if use_ema and self.model_ema is not None:
            self.model_ema.restore()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return {
            'loss': val_loss,
            'accuracy': val_acc
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if self.model_ema is not None:
            checkpoint['ema_state_dict'] = self.model_ema.shadow
        
        torch.save(checkpoint, filepath)
        print(f'💾 Checkpoint saved: {filepath}')

    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)

        total_start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1

            print(f"\n{'='*70}")
            print(f"EPOCH {self.current_epoch}/{self.config.num_epochs}")
            print(f"{'='*70}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate(use_ema=(self.model_ema is not None))

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # Print summary
            from utils import Utils
            print(f"\n📊 Epoch {self.current_epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
            print(f"  LR: {current_lr:.6f} | Time: {Utils.format_time(train_metrics['time'])}")

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0

                if self.config.save_best_only:
                    checkpoint_path = os.path.join(
                        self.config.checkpoint_dir,
                        f'best_model_acc_{self.best_val_acc:.2f}.pth'
                    )
                    self.save_checkpoint(checkpoint_path)
                print(f'🎯 New best validation acc: {self.best_val_acc:.2f}%')
            else:
                self.epochs_without_improvement += 1

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f'\n⚠️  Early stopping triggered after {self.current_epoch} epochs')
                print(f'   No improvement for {self.config.early_stopping_patience} epochs')
                break

        total_time = time.time() - total_start_time
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Total training time: {Utils.format_time(total_time)}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        return self.history

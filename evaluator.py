import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any

from config import Config

class Evaluator:

    def __init__(self, model: nn.Module, test_loader : DataLoader, config: Config):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.class_names = ['NORMAL', 'PNEUMONIA']

    def evaluate(self) -> Dict[str, Any]:
        print('\n' + '='*70)
        print('Model evaluation on test set')
        print('='*70)

        self.model.eval()
        self.model.to(self.config.device)

        all_preds = []
        all_labels = []
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                
                outputs = self.model(inputs)
                _,predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.* correct/total

        print(f'\n Classification report')
        print(classification_report(
            all_labels,
            all_preds,
            target_names = self.class_names,
            digits = 4
        ))

        cm = confusion_matrix(all_labels, all_preds)
        print("\n📈 Confusion Matrix:")
        print(f"{'':15} {'Pred NORMAL':>12} {'Pred PNEUMONIA':>15}")
        print(f"{'True NORMAL':15} {cm[0][0]:>12} {cm[0][1]:>15}")
        print(f"{'True PNEUMONIA':15} {cm[1][0]:>12} {cm[1][1]:>15}")
        
        print(f"\n✓ Test Accuracy: {accuracy:.2f}%")
        print(f"  Correct predictions: {correct}/{total}")
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'confusion_matrix': cm
        }
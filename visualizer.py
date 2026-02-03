import matplotlib.pyplot as plt
from typing import Dict, List

class Visualizer:

    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize = (15,10))

        epochs = range(1, len(history['train_loss']) + 1)

        #Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label = 'Train loss', linewidth = 2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label = 'Val loss', linewidth = 2)
        axes[0, 0].set_xlabel('Epoch', fontsize = 12)
        axes[0, 0].set_ylabel('Loss', fontsize = 12)
        axes[0, 0].set_title('Training & Validation Loss', fontsize = 14, fontweight = 'bold')
        axes[0, 0].legend(fontsize = 10)
        axes[0, 0].grid(True, alpha = 0.3)

        #Accuracy
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label = 'Train Acc', linewidth = 2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label = 'Val Acc', linewidth = 2)
        axes[0, 1].set_xlabel('Epoch', fontsize = 12)
        axes[0, 1].set_ylabel('Acc', fontsize = 12)
        axes[0, 1].set_title('Training & Validation Accuracy', fontsize = 14, fontweight = 'bold')
        axes[0, 1].legend(fontsize = 10)
        axes[0, 1].grid(True, alpha = 0.3)

        #Learning rate
        axes[1, 0].plot(epochs, history['learning_rates'], 'g-',  linewidth = 2)
        axes[1, 0].set_xlabel('Epoch', fontsize = 12)
        axes[1, 0].set_ylabel('Learning rate', fontsize = 12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize = 14, fontweight = 'bold')
        axes[1, 0].grid(True, alpha = 0.3)
        axes[1, 0].set_yscale('log')

        #Summary

        axes[1, 1].axis('off')

        summary_text = f"""
Training Summary
─────────────────────────────
Total Epochs: {len(epochs)}
Best Val Acc: {max(history['val_acc']):.2f}%
Final Train Acc: {history['train_acc'][-1]:.2f}%
Final Val Acc: {history['val_acc'][-1]:.2f}%
Final LR: {history['learning_rates'][-1]:.6f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                        verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")
        plt.close()



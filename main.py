import os

from config import Config
from utils import Utils
from data_loader import DataManager
from fine_tuning_model import PneumoniaCNN, FineTuningOptimizer
from trainer import Trainer
from evaluator import Evaluator
from visualizer import Visualizer

def main():

    
    print("\n" + "="*70)
    print("   PNEUMONIA DETECTION CNN - PRODUCTION PIPELINE")
    print("="*70)


    #Config init
    print("\n[STEP 1/7] Initializing configuration...")
    config = Config()
    config.validate()

    #Random seeds
    print('\n[STEP 2/7] Setting Random seeds...')
    Utils.set_seed(config.seed)

    #Setup data pipeline
    print('\n [STEP 3/7] Setting up data pipeline...')
    data_manager = DataManager(config)
    data_manager.load_datasets()
    train_loader, val_loader, test_loader = data_manager.create_dataloaders()

    #Building model
    print("\n[STEP 4/7] Building transfer learning model...")
    print("\n" + "="*70)
    print("TRANSFER LEARNING MODEL - RESNET50")
    print("="*70)

    datset_size = len(train_loader.dataset)

    if datset_size < 1000:
        freeze_strategy = 'all'
    elif datset_size <10000:
        freeze_strategy = 'partial'
    else:
        freeze_strategy ='none'

    model = PneumoniaCNN(config, freeze_strategy = freeze_strategy)
    param_counts = Utils.count_parameters(model)

    print(f"\n📊 Model Statistikası:")
    print(f"  Total parameters:      {param_counts['total']:>12,}")
    print(f"  Trainable parameters:  {param_counts['trainable']:>12,}")
    print(f"  Non-trainable (frozen):{param_counts['non_trainable']:>12,}")
    print(f"  Trainable ratio:       {param_counts['trainable']/param_counts['total']*100:>11.1f}%")

    optimizer = FineTuningOptimizer.create_optimizer(model, config)

    #Trainining model
    print("\n[STEP 5/7] Training model...")
    trainer = Trainer(model, train_loader,val_loader, config)
    trainer.optimizer = optimizer
    history = trainer.train()

    #Visuals
    print("\n[STEP 6/7] Visualizing training results...")
    plot_path = os.path.join(config.plot_dir, f'training_history_{Utils.get_timestamp()}.png')
    Visualizer.plot_training_history(history, plot_path)

    #Evaluation
    print("\n[STEP 7/7] Evaluating on test set...")
    evaluator = Evaluator(model, test_loader, config)
    test_results = evaluator.evaluate()

    #Completed
    print("\n" + "="*70)
    print("   PIPELINE COMPLETED SUCCESSFULLY! ✓")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"  Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"\nSaved Files:")
    print(f"  Model checkpoint: {config.checkpoint_dir}/")
    print(f"  Training plot: {plot_path}")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()

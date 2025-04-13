import argparse
import torch
import torch.nn as nn
import os
import random
import numpy as np
from datetime import datetime
import json
from src.dataset import get_data_loaders
from src.model import BallCounterCNN, SimplerBallCounterCNN
from src.train import train_model, evaluate_model, setup_training
from src.visualization import (
    plot_training_history, 
    plot_confusion_matrix,
    visualize_multiple_samples,
    visualize_filters,
    visualize_feature_maps,
    visualize_tsne
)
def save_args_to_json(args, filename='args.json',path=None):
    """Save command line arguments to a JSON file"""
    filename = filename if path is None else os.path.join(path,filename)
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"Arguments saved to {filename}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate ball counter CNN')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing class folders')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--model_type', type=str, default='simple', 
                        choices=['simple', 'complex'], help='Model architecture to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations after training')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save path for best model
    model_save_path = os.path.join(save_dir, 'best_model.pth')
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        num_samples_per_class=args.samples,
        binary=True,
        random_contrast=True,  # 启用随机明暗变化
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    if args.model_type == 'simple':
        model = SimplerBallCounterCNN(num_classes=10)
    else:
        model = BallCounterCNN(num_classes=10)
    
    # Print model summary
    #print(model)
    #if torch.cuda.device_count() > 1:
    #    print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    #    model = nn.DataParallel(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Setup training
    criterion, optimizer, scheduler = setup_training(
        model, learning_rate=args.learning_rate
    )
    plot_sav_path = save_dir+"/"
    
    # Train model
    print("Training model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader = test_loader,
        plot_sav_path = plot_sav_path,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=args.device,
        save_path=model_save_path
    )
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_loss, test_acc, all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=args.device
    )
    
    # Save test results
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
    
    # Plot training history
    
    plot_training_history(history,path=plot_sav_path)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds,path=plot_sav_path,class_names=10)
    save_args_to_json(args,path=plot_sav_path)

    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        
        # Get a batch of test images
        images, labels = next(iter(test_loader))
        
        # Visualize filters from first layer
        print("Visualizing filters...")
        if args.model_type == 'simple':
            visualize_filters(model, layer_name='conv1',path=plot_sav_path)
        else:
            visualize_filters(model, layer_name='conv1_1',path=plot_sav_path)
        
        # Visualize feature maps
        print("Visualizing feature maps...")
        visualize_feature_maps(model, images[0].unsqueeze(0), device=args.device,path=plot_sav_path)
        
        # Visualize grad-CAM for multiple samples
        print("Visualizing Grad-CAM...")
        visualize_multiple_samples(
            model=model,
            dataloader=test_loader,
            num_samples=10,
            layer_name='block3' if args.model_type == 'simple' else 'block4',
            device=args.device,
            path =plot_sav_path
        )
        
        # Visualize t-SNE embedding
        print("Visualizing t-SNE embedding...")
        visualize_tsne(model, test_loader, device=args.device,path =plot_sav_path)
    
    print(f"All results saved to {save_dir}")

if __name__ == "__main__":
    main()
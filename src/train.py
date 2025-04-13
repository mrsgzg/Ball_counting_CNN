import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import time
import copy
from tqdm import tqdm
import os
from src.visualization import (
    plot_training_history, 
    plot_confusion_matrix,
    visualize_multiple_samples,
    visualize_filters,
    visualize_feature_maps,
    visualize_tsne
)



def train_model(model, train_loader, test_loader,plot_sav_path, val_loader, criterion, optimizer, scheduler=None,
                num_epochs=25, device='cuda', save_path='model.pth'):
    """
    Train the CNN model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        save_path: Path to save the best model
        
    Returns:
        Trained model and training history
    """
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Move model to device
    model = model.to(device)
    
    # Variables to track best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        plot_sav_path_use = plot_sav_path+'/epoch'+str(epoch)
        os.makedirs(plot_sav_path_use, exist_ok=True)
        torch.save(model.state_dict(), plot_sav_path_use+'/model.pth')
        print(f'Epoch:{epoch} model saved')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        visualize_multiple_samples(
            model=model,
            dataloader=test_loader,
            num_samples=10,
            layer_name='block3',
            device='cuda',
            path =plot_sav_path_use+'/'+'block3')
        
        visualize_multiple_samples(
            model=model,
            dataloader=test_loader,
            num_samples=10,
            layer_name='block2',
            device='cuda',
            path =plot_sav_path_use+'/'+'block2')
        
        visualize_multiple_samples(
            model=model,
            dataloader=test_loader,
            num_samples=10,
            layer_name='block1',
            device='cuda',
            path =plot_sav_path_use+'/'+'block1')

        test_loss, test_acc, all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device='cuda')

        plot_confusion_matrix(all_labels, all_preds,path=plot_sav_path_use+'/',class_names=10)
        visualize_tsne(model, test_loader, device='cuda',path =plot_sav_path_use+'/')
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Adjust learning rate if scheduler is provided
                if scheduler is not None:
                    scheduler.step(epoch_loss)
                
                # Save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)
                    print(f'New best model saved with accuracy: {best_acc:.4f}')
            
    # Load best model weights
    history_df = pd.DataFrame(history)
    csv_save_path = plot_sav_path_use+'/'+'history.csv'
    history_df.to_csv(csv_save_path, index=False)
    model.load_state_dict(best_model_wts)
    
    return model, history

def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Test loss, accuracy, all predictions, and all ground truth labels
    """
    model.eval()  # Set model to evaluate mode
    model = model.to(device)
    
    # Track metrics
    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []
    
    # Iterate over data
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            
            # Store predictions and labels for detailed analysis
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc.item(), np.array(all_preds), np.array(all_labels)

def setup_training(model, learning_rate=0.001, weight_decay=1e-4):
    """
    Set up training configuration
    
    Args:
        model: PyTorch model
        learning_rate: Initial learning rate
        weight_decay: L2 regularization parameter
        
    Returns:
        Loss criterion, optimizer, and scheduler
    """
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
        verbose=True, min_lr=1e-6
    )
    
    return criterion, optimizer, scheduler
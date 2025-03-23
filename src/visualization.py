import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import seaborn as sns

def plot_training_history(history):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    if class_names is None:
        class_names = [f'{i+1} Ball(s)' for i in range(5)]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print(classification_report(y_true, y_pred, target_names=class_names))

def generate_gradcam(model, image, target_class=None, layer_name='block4'):
    """
    Generate Grad-CAM heatmap for model interpretation
    
    Args:
        model: Trained PyTorch model
        image: Input image tensor (C,H,W)
        target_class: Target class index (if None, uses predicted class)
        layer_name: Layer to use for Grad-CAM ('block1', 'block2', 'block3', 'block4')
        
    Returns:
        Original image, Grad-CAM heatmap, and superimposed visualization
    """
    model.eval()
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    # Get device
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Forward pass
    image.requires_grad_()
    
    # Get activations for the target layer
    activations = model.get_activation_maps(image, layer_name)
    
    # Forward pass through the rest of the model
    output = model(image)
    
    # If no target class provided, use the predicted class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # One-hot encode the target class
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    
    # Backward pass
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Get gradients
    gradients = image.grad
    
    # Global average pooling of the gradients
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    
    # Weight the activations by the gradients
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    
    # Apply ReLU to the CAM
    cam = F.relu(cam)
    
    # Normalize the CAM
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Convert to numpy and resize to original image size
    cam = cam.detach().cpu().numpy()[0, 0]
    
    # Resize to image size
    cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Get original image as numpy array
    orig_img = image[0].detach().cpu().numpy()
    orig_img = np.transpose(orig_img, (1, 2, 0))
    
    # Normalize image to 0-255
    if orig_img.max() <= 1:
        orig_img = orig_img * 255
    orig_img = orig_img.astype(np.uint8)
    
    # If grayscale, convert to RGB
    if orig_img.shape[2] == 1:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
    
    # Superimpose heatmap on original image
    superimposed = heatmap * 0.5 + orig_img
    superimposed = superimposed.astype(np.uint8)
    
    return orig_img, heatmap, superimposed

def visualize_multiple_samples(model, dataloader, class_names=None, num_samples=5, layer_name='block4', device='cuda'):
    """
    Visualize Grad-CAM for multiple samples
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader to get samples from
        class_names: List of class names
        num_samples: Number of samples to visualize
        layer_name: Layer to use for Grad-CAM
        device: Device to run model on
    """
    if class_names is None:
        class_names = [f'{i+1} Ball(s)' for i in range(5)]
    
    model = model.to(device)
    model.eval()
    
    # Get samples
    images = []
    labels = []
    
    for img, lbl in dataloader:
        images.extend(img)
        labels.extend(lbl)
        
        if len(images) >= num_samples:
            break
    
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i, (img, lbl) in enumerate(zip(images, labels)):
        # Generate Grad-CAM
        img = img.to(device)
        orig_img, heatmap, superimposed = generate_gradcam(model, img, target_class=lbl, layer_name=layer_name)
        
        # Convert tensors to numpy arrays for plotting
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.shape[2] == 1:  # If grayscale, repeat channels
                img_np = np.repeat(img_np, 3, axis=2)
        else:
            img_np = img
        
        # Plot original image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Original: {class_names[lbl]}')
        axes[i, 0].axis('off')
        
        # Plot heatmap
        axes[i, 1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title('Grad-CAM Heatmap')
        axes[i, 1].axis('off')
        
        # Plot superimposed
        axes[i, 2].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title('Superimposed')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png')
    plt.show()

def visualize_filters(model, layer_name='conv1', num_filters=16):
    """
    Visualize CNN filters for a specific layer
    
    Args:
        model: Trained PyTorch model
        layer_name: Name of the layer to visualize
        num_filters: Number of filters to display
    """
    # Get the layer weights
    if layer_name == 'conv1':
        filters = model.conv1.weight.data.cpu().numpy()
    elif layer_name == 'conv2':
        filters = model.conv2.weight.data.cpu().numpy()
    elif layer_name == 'conv3':
        filters = model.conv3.weight.data.cpu().numpy()
    else:
        raise ValueError(f"Unsupported layer name: {layer_name}")
    
    # Limit the number of filters to display
    num_filters = min(num_filters, filters.shape[0])
    
    # Plot the filters
    fig, axes = plt.subplots(4, num_filters//4, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(num_filters):
        # Get filter
        f = filters[i, 0]  # Take first input channel for visualization
        
        # Normalize filter
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        
        # Plot filter
        axes[i].imshow(f, cmap='viridis')
        axes[i].set_title(f'Filter {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{layer_name}_filters.png')
    plt.show()

def visualize_feature_maps(model, image, layer_names=None, device='cuda'):
    """
    Visualize feature maps for a given image
    
    Args:
        model: Trained PyTorch model
        image: Input image tensor
        layer_names: List of layer names to visualize
        device: Device to run model on
    """
    if layer_names is None:
        layer_names = ['block1', 'block2', 'block3', 'block4']
    
    model = model.to(device)
    model.eval()
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    # Forward pass with feature extraction
    with torch.no_grad():
        features = model(image, return_features=True)
    
    # Create figure
    num_layers = len(layer_names)
    fig, axes = plt.subplots(num_layers, 8, figsize=(20, 5 * num_layers))
    
    # If only one layer, add dimension to axes
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for i, layer_name in enumerate(layer_names):
        # Get feature maps
        feature_maps = features[layer_name]
        
        # Convert to numpy
        feature_maps = feature_maps.cpu().numpy()
        
        # Select 8 channels to display
        num_channels = min(8, feature_maps.shape[1])
        step = max(1, feature_maps.shape[1] // 8)
        
        for j in range(num_channels):
            channel_idx = j * step
            
            # Get feature map
            feature_map = feature_maps[0, channel_idx]
            
            # Normalize feature map
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            # Plot feature map
            axes[i, j].imshow(feature_map, cmap='viridis')
            axes[i, j].set_title(f'Channel {channel_idx}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_maps.png')
    plt.show()

def visualize_tsne(model, dataloader, device='cuda', perplexity=30, n_iter=1000):
    """
    Visualize t-SNE embedding of features from the penultimate layer
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for the dataset
        device: Device to run model on
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
    """
    model = model.to(device)
    model.eval()
    
    # Collect features and labels
    features = []
    labels = []
    
    with torch.no_grad():
        for img, lbl in dataloader:
            img = img.to(device)
            
            # Get features from penultimate layer - simplified approach
            if hasattr(model, 'forward'):
                # Use the model's built-in feature extraction
                feature_dict = model(img, return_features=True)
                batch_features = feature_dict['penultimate'].cpu().numpy()
            else:
                # For models without return_features, use a step-by-step approach
                # This replaces the complex nested function call with more readable code
                x = img
                
                # Block 1
                x = F.relu(model.bn1_1(model.conv1_1(x)))
                x = F.relu(model.bn1_2(model.conv1_2(x)))
                x = model.pool1(x)
                x = model.dropout1(x)
                
                # Block 2
                x = F.relu(model.bn2_1(model.conv2_1(x)))
                x = F.relu(model.bn2_2(model.conv2_2(x)))
                x = model.pool2(x)
                x = model.dropout2(x)
                
                # Block 3
                x = F.relu(model.bn3_1(model.conv3_1(x)))
                x = F.relu(model.bn3_2(model.conv3_2(x)))
                x = model.pool3(x)
                x = model.dropout3(x)
                
                # Block 4
                x = F.relu(model.bn4_1(model.conv4_1(x)))
                x = F.relu(model.bn4_2(model.conv4_2(x)))
                x = model.pool4(x)
                
                # Flatten and get features from fc1
                x = torch.flatten(x, 1)
                x = model.fc1(x)
                batch_features = x.cpu().numpy()
            
            features.append(batch_features)
            labels.append(lbl.numpy())
    
    # Concatenate all features and labels
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Computing t-SNE on {features.shape[0]} samples...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_embedded = tsne.fit_transform(features)
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    
    for i in range(5):  # 5 classes (1-5 balls)
        plt.scatter(
            features_embedded[labels == i, 0],
            features_embedded[labels == i, 1],
            label=f'{i+1} Ball(s)',
            alpha=0.7
        )
    
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tsne_visualization.png')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import seaborn as sns

def plot_training_history(history,path):
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
    path = path+"training_history.png"
    plt.savefig(path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None,path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    #if class_names is None:
    class_names = [f'{i+1} Ball(s)' for i in range(class_names)]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    path = path+"confusion_matrix.png"
    plt.savefig(path)
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



def visualize_multiple_samples(model, dataloader, num_samples=5, layer_name='block4', device='cuda', path=None):
    """
    Visualize Grad-CAM for multiple samples, handling any number of balls
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader to get samples from
        num_samples: Number of samples to visualize
        layer_name: Layer to use for Grad-CAM
        device: Device to run model on
        path: Path to save the visualization
    """
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
    
    # Convert labels to list of integers if they're tensors
    if hasattr(labels[0], 'item'):
        labels = [label.item() for label in labels]
    
    print(f"Sample labels: {labels}")
    
    # Determine the number of classes from the data
    # Add 1 to max label to account for 0-indexing
    max_label = max(labels)
    num_classes = max_label + 1
    
    # Create class names dynamically based on the max number of balls
    class_names = [f'{i} Ball(s)' for i in range(num_classes)]
    print(f'Class names: {class_names}')

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # Handle case where there's only one sample
    if num_samples == 1:
        axes = [axes]
    
    for i, (img, lbl) in enumerate(zip(images, labels)):
        # Generate Grad-CAM
        img_tensor = img.to(device).unsqueeze(0) if img.dim() == 3 else img.to(device)
        orig_img, heatmap, superimposed = generate_gradcam(model, img_tensor, target_class=lbl, layer_name=layer_name)
        
        # Convert tensors to numpy arrays for plotting
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
            if img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
                if img_np.shape[2] == 1:  # If grayscale, repeat channels
                    img_np = np.repeat(img_np, 3, axis=2)
        else:
            img_np = img
        
        # Normalize image for display if needed
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Make sure label is in bounds
        if lbl >= len(class_names):
            class_label = f'{lbl} Ball(s)'
        else:
            class_label = class_names[lbl]
        
        # Plot original image
        axes[i][0].imshow(img_np)
        axes[i][0].set_title(f'Original: {class_label}')
        axes[i][0].axis('off')
        
        # Plot heatmap
        axes[i][1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        axes[i][1].set_title('Grad-CAM Heatmap')
        axes[i][1].axis('off')
        
        # Plot superimposed
        axes[i][2].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        axes[i][2].set_title('Superimposed')
        axes[i][2].axis('off')
    
    plt.tight_layout()
    if path:
        plt.savefig(path + "gradcam_visualization.png")
    plt.show()
'''
def visualize_multiple_samples(model, dataloader, num_samples=5, layer_name='block4', device='cuda', path=None):
    """
    Visualize Grad-CAM for multiple samples
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader to get samples from
        num_samples: Number of samples to visualize
        layer_name: Layer to use for Grad-CAM
        device: Device to run model on
        path: Path to save the visualization
    """
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
    print(f"***{labels}***")
    # Determine the number of unique labels
    unique_labels = sorted(set(labels))
    num_classes = len(unique_labels)
    class_names = [f'{i+1} Ball(s)' for i in range(num_classes)]
    print(f'***{class_names}***')

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 10 * num_samples))
    
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
    if path:
        plt.savefig(path + "gradcam_visualization.png")
    plt.show()
'''
'''
def visualize_multiple_samples(model, dataloader, class_names=None, num_samples=5, layer_name='block4', device='cuda',path=None):
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
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 10 * num_samples))
    
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
    path = path+"gradcam_visualization.png"
    plt.savefig(path)
    plt.show()
'''
def visualize_filters(model, layer_name='conv1', num_filters=16, path=None):
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
    path = path+str(layer_name)+"_filters.png"
    plt.savefig(path)
    plt.show()

def visualize_feature_maps(model, image, layer_names=None, device='cuda',path=None):
    """
    Visualize feature maps for a given image
    
    Args:
        model: Trained PyTorch model
        image: Input image tensor
        layer_names: List of layer names to visualize
        device: Device to run model on
    """
    # Default layer names
    if layer_names is None:
        # For SimplerBallCounterCNN
        if hasattr(model, 'gap'):
            layer_names = ['block1', 'block2', 'block3']
        # For BallCounterCNN
        else:
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
    
    # Check which layers are actually available
    available_layers = list(features.keys())
    print(f"Available feature layers: {available_layers}")
    
    # Filter layer_names to only include available layers
    valid_layer_names = [name for name in layer_names if name in available_layers]
    
    if not valid_layer_names:
        print(f"Warning: None of the requested layers {layer_names} are available.")
        print(f"Using available layers instead.")
        # Use the first few available feature layers, excluding 'output'
        valid_layer_names = [name for name in available_layers if name != 'output'][:3]
    
    print(f"Visualizing layers: {valid_layer_names}")
    
    # Create figure
    num_layers = len(valid_layer_names)
    fig, axes = plt.subplots(num_layers, 8, figsize=(20, 5 * num_layers))
    
    # If only one layer, add dimension to axes
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for i, layer_name in enumerate(valid_layer_names):
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
            axes[i, j].set_title(f'{layer_name} - Channel {channel_idx}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    path = path+"feature_maps.png"
    plt.savefig(path)
    plt.show()

def visualize_tsne(model, dataloader, device='cuda', perplexity=30, n_iter=1000, path=None):
    """
    Visualize t-SNE embedding of features from the penultimate layer
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for the dataset
        device: Device to run model on
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        path: Path to save the visualization
    """
    model = model.to(device)
    model.eval()
    
    # Collect features and labels
    features = []
    labels = []
    
    with torch.no_grad():
        # Get one batch to identify available feature layers
        for img, lbl in dataloader:
            img = img.to(device)
            feature_dict = model(img, return_features=True)
            available_features = list(feature_dict.keys())
            print(f"Available features: {available_features}")
            break
        
        # Determine which feature to use (penultimate layer)
        feature_to_use = None
        # Check for specific feature names
        if 'penultimate' in available_features:
            feature_to_use = 'penultimate'
        elif 'pooled' in available_features:
            feature_to_use = 'pooled'
        elif 'gap' in available_features:
            feature_to_use = 'gap'
        else:
            # Use the layer before 'output' as fallback
            non_output_features = [f for f in available_features if f != 'output']
            if non_output_features:
                feature_to_use = non_output_features[-1]  # Last feature before output
        
        if not feature_to_use:
            print("Could not find a suitable feature layer for t-SNE visualization")
            return
            
        print(f"Using '{feature_to_use}' for t-SNE visualization")
            
        # Now extract features from all batches
        for img, lbl in dataloader:
            img = img.to(device)
            
            # Extract features
            feature_dict = model(img, return_features=True)
            batch_features = feature_dict[feature_to_use].cpu().numpy()
            
            # For SimplerBallCounterCNN, the feature might be 4D with dimensions [batch, channels, 1, 1]
            # We need to flatten it to 2D [batch, features]
            if len(batch_features.shape) > 2:
                batch_features = batch_features.reshape(batch_features.shape[0], -1)
            
            features.append(batch_features)
            
            # Convert labels to numpy array if they're tensors
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.cpu().numpy()
            labels.append(lbl)
    
    # Concatenate all features and labels
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Computing t-SNE on {features.shape[0]} samples...")
    print(f"Feature shape: {features.shape}")
    
    # Get unique classes
    unique_labels = np.unique(labels)
    max_label = np.max(labels)
    num_classes = max_label + 1  # +1 because labels are typically 0-indexed
    
    print(f"Dataset contains {len(unique_labels)} unique classes from 0 to {max_label}")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    features_embedded = tsne.fit_transform(features)
    
    # Plot t-SNE
    plt.figure(figsize=(12, 10))
    
    # Use a colormap that can handle many classes
    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'tab20')
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_embedded[mask, 0],
            features_embedded[mask, 1],
            label=f'{label} Ball(s)',
            color=cmap(i % cmap.N),
            alpha=0.7
        )
    
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if path is not None:
        save_path = path + "tsne_visualization.png" if not path.endswith('.png') else path
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()
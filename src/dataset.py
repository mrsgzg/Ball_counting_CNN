import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class BallDataset(Dataset):
    """Ball counter dataset for loading and preprocessing images"""
    
    def __init__(self, image_paths, labels, transform=None, binary=True, lower_threshold=200, upper_threshold=255,
                 random_contrast=True):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (integer class indices)
            transform: Optional transforms to apply
            binary: Whether to convert images to binary
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.binary = binary
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.random_contrast = random_contrast
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
         # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size (320, 240)
        gray = cv2.resize(gray, (320, 240))
        
        # Convert to binary if needed
        if self.binary:
            # Apply thresholding to isolate white objects
            _, binary = cv2.threshold(gray, self.lower_threshold, self.upper_threshold, cv2.THRESH_BINARY)
            
            # Clean up the binary image using morphological operations
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            if self.random_contrast:
                # 为每个连通区域（球）分配不同的灰度值
                # 首先，找到所有连通区域
                num_labels, labels = cv2.connectedComponents(binary)
                
                # 创建新的图像，初始值为0（黑色背景）
                contrast_image = np.zeros_like(binary)
                
                # 为每个标签（除了背景，背景是0）分配一个随机灰度值
                for label in range(1, num_labels):
                    # 为每个球生成不同的随机灰度值
                    # 使用更合理的范围，例如30到200
                    gray_value = random.randint(30, 200)
                    
                    # 将该标签的所有像素设置为这个灰度值
                    contrast_image[labels == label] = gray_value
                
                # 使用这个具有不同灰度级别的图像
                image = contrast_image
                '''
                # 可选：打印调试信息
                if random.random() < 0.01:  # 只打印1%的样本，避免输出过多
                    print(f"形状: {image.shape}")
                    print(f"数据类型: {image.dtype}")
                    print(f"最小值: {image.min()}, 最大值: {image.max()}")
                    print(f"发现 {num_labels-1} 个球，灰度范围: 30-200")
                '''
            else:
                image = binary
        else:
            image = gray


        '''
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size (320, 240)
        image = cv2.resize(image, (320, 240))
        
        # Convert to binary if needed
        if self.binary:
            # Apply adaptive thresholding for better handling of lighting variations
            image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        '''
        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(image)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: convert to tensor and normalize
            image = transforms.ToTensor()(image)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label

def load_dataset(data_dir, num_samples_per_class=500):
    """
    Load dataset from directory structure with balanced sampling
    
    Args:
        data_dir: Root directory containing class folders (1-5)
        num_samples_per_class: Number of samples to select from each class
        
    Returns:
        List of image paths and corresponding labels
    """
    image_paths = []
    labels = []
    
    # For each class folder (1-5)
    for class_idx in range(1, 11):
        class_dir = os.path.join(data_dir, str(class_idx))
        if not os.path.exists(class_dir):
            raise ValueError(f"Class directory {class_dir} not found")
        
        # Get list of image files in this class folder
        files = []
        for root, dirs, files_names in os.walk(class_dir):
            for file_name in files_names:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    files.append(os.path.join(root, file_name)) 
                    #print(f"Adding file{file_name}")
                    #print(f"Adding root{root}")


        #files = [f for f in os.listdir(class_dir) 
        #        if os.path.isfile(os.path.join(class_dir, f)) and 
        #        f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sample files randomly
        if len(files) >= num_samples_per_class:
            selected_files = random.sample(files, num_samples_per_class)
        else:
            selected_files = files
            print(f"Warning: Class {class_idx} has only {len(files)} images (requested {num_samples_per_class})")
        
        # Add selected files to dataset
        for img_path in selected_files:
           
            image_paths.append(img_path)
            labels.append(class_idx - 1)  # Use 0-based indexing for classes
    
    return image_paths, labels

def get_data_loaders(data_dir, num_samples_per_class=500, batch_size=32, 
                     val_split=0.15, test_split=0.15, seed=42,
                     binary=True, random_contrast=True):
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir: Root directory with class folders
        num_samples_per_class: Number of samples per class
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        seed: Random seed for reproducibility
        binary: Whether to use binary images
        random_brightness: Whether to apply random brightness to training images
        brightness_range_train: Range of brightness values for training (min, max)
        brightness_val: Fixed brightness value for validation and test sets
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    image_paths, labels = load_dataset(data_dir, num_samples_per_class)
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_split, stratify=labels, random_state=seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_split/(1-test_split),
        stratify=y_temp, 
        random_state=seed
    )
    
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)} images")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = BallDataset(
        X_train, y_train, 
        transform=train_transform, 
        binary=binary,
        random_contrast=random_contrast,
    )
    
    # For validation and test, use fixed brightness for consistency
    val_dataset = BallDataset(
        X_val, y_val, 
        transform=val_transform, 
        binary=binary,
        random_contrast=random_contrast
    )
    
    test_dataset = BallDataset(
        X_test, y_test, 
        transform=val_transform, 
        binary=binary,
        random_contrast=random_contrast
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
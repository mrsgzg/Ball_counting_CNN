import os
import pytest
import shutil
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import BallDataset, load_dataset, get_data_loaders

class TestDataset:
    """Tests for the dataset module"""
    
    @pytest.fixture
    def setup_test_data(self, tmp_path):
        """Create a temporary directory with test data"""
        # Create directory structure
        data_dir = tmp_path / "data"
        for i in range(1, 6):  # Create folders for 1-5 balls
            class_dir = data_dir / str(i)
            class_dir.mkdir(parents=True)
            
            # Create dummy images
            for j in range(10):  # 10 images per class for testing
                img = np.zeros((480, 640), dtype=np.uint8)
                # Add some circles to represent balls
                for k in range(i):  # Draw i balls
                    center_x = np.random.randint(50, 590)
                    center_y = np.random.randint(50, 430)
                    radius = np.random.randint(20, 40)
                    cv = np.sqrt((np.indices((480, 640))[0] - center_y)**2 + 
                               (np.indices((480, 640))[1] - center_x)**2)
                    img[cv < radius] = 255
                
                # Save image
                img_path = class_dir / f"image_{j}.png"
                Image.fromarray(img).save(img_path)
        
        return data_dir
    
    def test_load_dataset(self, setup_test_data):
        """Test loading dataset from directory"""
        data_dir = setup_test_data
        image_paths, labels = load_dataset(data_dir, num_samples_per_class=5)
        
        # Check number of images and labels
        assert len(image_paths) == 25  # 5 per class, 5 classes
        assert len(labels) == 25
        
        # Check distribution of labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        assert len(unique_labels) == 5
        for count in counts:
            assert count == 5
    
    def test_dataset_class(self, setup_test_data):
        """Test BallDataset class"""
        data_dir = setup_test_data
        image_paths, labels = load_dataset(data_dir, num_samples_per_class=5)
        
        # Create dataset
        dataset = BallDataset(image_paths, labels, binary=True)
        
        # Check dataset length
        assert len(dataset) == 25
        
        # Get an item
        img, label = dataset[0]
        
        # Check image shape and type
        assert img.shape[0] == 1  # Channel dimension
        assert img.shape[1] == 240  # Height
        assert img.shape[2] == 320  # Width
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        
        # Check label type
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        
        # Test dataset with transforms
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        dataset_with_transform = BallDataset(image_paths, labels, transform=transform)
        img, label = dataset_with_transform[0]
        
        # Check image shape and type with transforms
        assert img.shape[0] == 1  # Channel dimension
        assert img.shape[1] == 240  # Height
        assert img.shape[2] == 320  # Width
    
    def test_data_loaders(self, setup_test_data):
        """Test data loader creation"""
        data_dir = setup_test_data
        
        # Create data loaders
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir, num_samples_per_class=5, batch_size=2
        )
        
        # Check that loaders are created
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Check batch from train loader
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] == 2  # Batch size
        assert batch_x.shape[1] == 1  # Channel
        assert batch_x.shape[2] == 240  # Height
        assert batch_x.shape[3] == 320  # Width
        assert batch_y.shape[0] == 2  # Batch size
        
        # Check that data augmentation works (train only)
        # We can't directly test randomness, but we can ensure the code runs
        for _ in range(3):
            batch_x, batch_y = next(iter(train_loader))
import os
import pytest
import torch
import shutil
import numpy as np
from PIL import Image
import cv2

from src.dataset import get_data_loaders
from src.model import SimplerBallCounterCNN, BallCounterCNN
from src.train import train_model, evaluate_model, setup_training
from src.visualization import generate_gradcam, visualize_filters, visualize_feature_maps

class TestIntegration:
    """End-to-end tests for training and inference pipeline"""
    
    @pytest.fixture
    def setup_test_data(self, tmp_path):
        """Create a temporary directory with test data"""
        # Create directory structure
        data_dir = tmp_path / "data"
        for i in range(1, 6):  # Create folders for 1-5 balls
            class_dir = data_dir / str(i)
            class_dir.mkdir(parents=True)
            
            # Create dummy images
            for j in range(20):  # 20 images per class for small integration test
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
    
    def test_small_training_loop(self, setup_test_data):
        """Test a small training loop to ensure integration works"""
        data_dir = setup_test_data
        
        # Create data loaders with small batch size
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir, num_samples_per_class=10, batch_size=4
        )
        
        # Create model
        model = SimplerBallCounterCNN(num_classes=5)
        
        # Setup training
        criterion, optimizer, scheduler = setup_training(
            model, learning_rate=0.001
        )
        
        # Train for just 2 epochs to verify everything works
        device = 'cpu'  # Use CPU for testing
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=2,
            device=device,
            save_path='test_model.pth'
        )
        
        # Check history was recorded
        assert 'train_loss' in history
        assert 'train_acc' in history
        assert 'val_loss' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) == 2
        
        # Clean up
        if os.path.exists('test_model.pth'):
            os.remove('test_model.pth')
    
    def test_evaluation(self, setup_test_data):
        """Test model evaluation"""
        data_dir = setup_test_data
        
        # Create data loaders with small batch size
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir, num_samples_per_class=10, batch_size=4
        )
        
        # Create and do a single forward pass to initialize model
        model = SimplerBallCounterCNN(num_classes=5)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Evaluate model
        device = 'cpu'  # Use CPU for testing
        test_loss, test_acc, all_preds, all_labels = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        # Check outputs
        assert isinstance(test_loss, float)
        assert isinstance(test_acc, float)
        assert isinstance(all_preds, np.ndarray)
        assert isinstance(all_labels, np.ndarray)
        assert len(all_preds) == len(all_labels)
    
    def test_visualization_pipeline(self, setup_test_data):
        """Test visualization functions"""
        data_dir = setup_test_data
        
        # Create data loaders with minimal samples
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir, num_samples_per_class=5, batch_size=1
        )
        
        # Create model
        model = SimplerBallCounterCNN(num_classes=5)
        device = 'cpu'  # Use CPU for testing
        model = model.to(device)
        
        # Get a sample image
        image, label = next(iter(test_loader))
        image = image.to(device)
        
        # Test Grad-CAM generation
        orig_img, heatmap, superimposed = generate_gradcam(
            model=model,
            image=image,
            target_class=label.item(),
            layer_name='block3'
        )
        
        # Check output types and shapes
        assert isinstance(orig_img, np.ndarray)
        assert isinstance(heatmap, np.ndarray)
        assert isinstance(superimposed, np.ndarray)
        
        # Test filter visualization
        try:
            visualize_filters(model, layer_name='conv1', num_filters=4)
            # If this runs without error, we count it as a pass
            filter_test_passed = True
        except Exception as e:
            filter_test_passed = False
        
        assert filter_test_passed
        
        # Test feature map visualization
        try:
            visualize_feature_maps(model, image, layer_names=['block1', 'block2'], device=device)
            # If this runs without error, we count it as a pass
            feature_map_test_passed = True
        except Exception as e:
            feature_map_test_passed = False
        
        assert feature_map_test_passed
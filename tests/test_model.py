import pytest
import torch
import numpy as np

from src.model import BallCounterCNN, SimplerBallCounterCNN

class TestModel:
    """Tests for the model architectures"""
    
    def test_simple_model_init(self):
        """Test initialization of the simpler model"""
        model = SimplerBallCounterCNN(num_classes=5)
        
        # Check model type
        assert isinstance(model, SimplerBallCounterCNN)
        
        # Check number of output classes
        assert model.fc.out_features == 5
        
        # Check model layers
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'conv3')
        assert hasattr(model, 'gap')
        assert hasattr(model, 'fc')
    
    def test_complex_model_init(self):
        """Test initialization of the more complex model"""
        model = BallCounterCNN(num_classes=5)
        
        # Check model type
        assert isinstance(model, BallCounterCNN)
        
        # Check number of output classes
        assert model.fc2.out_features == 5
        
        # Check model layers
        assert hasattr(model, 'conv1_1')
        assert hasattr(model, 'conv1_2')
        assert hasattr(model, 'conv2_1')
        assert hasattr(model, 'conv2_2')
        assert hasattr(model, 'conv3_1')
        assert hasattr(model, 'conv3_2')
        assert hasattr(model, 'conv4_1')
        assert hasattr(model, 'conv4_2')
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
    
    def test_simple_model_forward(self):
        """Test forward pass of the simpler model"""
        model = SimplerBallCounterCNN(num_classes=5)
        
        # Create dummy input (batch_size=2, channels=1, height=240, width=320)
        x = torch.randn(2, 1, 240, 320)
        
        # Test regular forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (2, 5)
        
        # Test forward pass with feature extraction
        features = model(x, return_features=True)
        
        # Check feature dictionary
        assert isinstance(features, dict)
        assert 'block1' in features
        assert 'block2' in features
        assert 'block3' in features
        assert 'output' in features
        
        # Check feature shapes
        # Block 1: After pooling, size should be reduced by factor of 2
        # Starting with 320x240, after stride 2 conv -> 160x120, after pool -> 80x60
        assert features['block1'].shape == (2, 32, 80, 60)
        
        # Block 2: After another pooling, size should be 40x30
        assert features['block2'].shape == (2, 64, 40, 30)
        
        # Block 3: After another pooling, size should be 20x15
        assert features['block3'].shape == (2, 128, 20, 15)
        
        # Output should match regular forward pass
        assert torch.all(torch.eq(features['output'], output))
    
    def test_complex_model_forward(self):
        """Test forward pass of the more complex model"""
        model = BallCounterCNN(num_classes=5)
        
        # Create dummy input (batch_size=2, channels=1, height=240, width=320)
        x = torch.randn(2, 1, 240, 320)
        
        # Test regular forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (2, 5)
        
        # Test forward pass with feature extraction
        features = model(x, return_features=True)
        
        # Check feature dictionary
        assert isinstance(features, dict)
        assert 'block1' in features
        assert 'block2' in features
        assert 'block3' in features
        assert 'block4' in features
        assert 'penultimate' in features
        assert 'output' in features
        
        # Check feature shapes
        # Block 1: After pooling, size should be reduced by factor of 2
        assert features['block1'].shape[1] == 32  # 32 channels
        
        # Block 2: After another pooling
        assert features['block2'].shape[1] == 64  # 64 channels
        
        # Block 3: After another pooling
        assert features['block3'].shape[1] == 128  # 128 channels
        
        # Block 4: After another pooling
        assert features['block4'].shape[1] == 256  # 256 channels
        
        # Penultimate should be 1D feature vector
        assert len(features['penultimate'].shape) == 2
        assert features['penultimate'].shape[0] == 2  # Batch size
        assert features['penultimate'].shape[1] == 512  # Feature dimension
        
        # Output should match regular forward pass
        assert torch.all(torch.eq(features['output'], output))
    
    def test_get_activation_maps(self):
        """Test activation map extraction"""
        # Test for simple model
        simple_model = SimplerBallCounterCNN(num_classes=5)
        
        # Create dummy input
        x = torch.randn(1, 1, 240, 320)
        
        # Get activation maps for different layers
        activations_block1 = simple_model.get_activation_maps(x, 'block1')
        activations_block2 = simple_model.get_activation_maps(x, 'block2')
        activations_block3 = simple_model.get_activation_maps(x, 'block3')
        
        # Check activation shapes
        assert activations_block1.shape[1] == 32  # 32 channels in block1
        assert activations_block2.shape[1] == 64  # 64 channels in block2
        assert activations_block3.shape[1] == 128  # 128 channels in block3
        
        # Test for complex model
        complex_model = BallCounterCNN(num_classes=5)
        
        # Get activation maps for different layers
        activations_block1 = complex_model.get_activation_maps(x, 'block1')
        activations_block4 = complex_model.get_activation_maps(x, 'block4')
        
        # Check activation shapes
        assert activations_block1.shape[1] == 32  # 32 channels in block1
        assert activations_block4.shape[1] == 256  # 256 channels in block4
        
        # Test invalid layer name
        with pytest.raises(ValueError):
            complex_model.get_activation_maps(x, 'invalid_layer')
    
    def test_model_with_different_input_sizes(self):
        """Test model with different input sizes"""
        simple_model = SimplerBallCounterCNN(num_classes=5)
        
        # Test with standard size
        x1 = torch.randn(1, 1, 240, 320)
        output1 = simple_model(x1)
        assert output1.shape == (1, 5)
        
        # Test with smaller size
        x2 = torch.randn(1, 1, 120, 160)
        output2 = simple_model(x2)
        assert output2.shape == (1, 5)
        
        # The complex model uses flattening which depends on specific input dimensions
        # So we'll just test it with the standard size
        complex_model = BallCounterCNN(num_classes=5)
        output3 = complex_model(x1)
        assert output3.shape == (1, 5)
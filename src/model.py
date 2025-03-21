import torch
import torch.nn as nn
import torch.nn.functional as F

class BallCounterCNN(nn.Module):
    """
    CNN model for counting balls in images with interpretability in mind
    """
    
    def __init__(self, num_classes=5):
        super(BallCounterCNN, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Block 4 (final conv block)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate input size for the first fully connected layer
        # Input: 320x240 -> After 4 pooling layers: 320/16 x 240/16 = 20x15
        fc_input_size = 256 * 15 * 20
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        block1_features = x
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        block2_features = x
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        block3_features = x
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        block4_features = x
        x = self.pool4(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        penultimate_features = x
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        # If requested, return intermediate features for visualization
        if return_features:
            return {
                'block1': block1_features,
                'block2': block2_features,
                'block3': block3_features,
                'block4': block4_features,
                'penultimate': penultimate_features,
                'output': x
            }
        
        return x
    
    def get_activation_maps(self, x, layer_name):
        """
        Get activation maps for a specific layer
        
        Args:
            x: Input tensor
            layer_name: Name of the layer to extract activations from
                (one of 'block1', 'block2', 'block3', 'block4')
                
        Returns:
            Activation tensor
        """
        activations = {}
        
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        activations['block1'] = x
        x = self.pool1(x)
        x = self.dropout1(x)
        
        if layer_name == 'block1':
            return activations['block1']
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        activations['block2'] = x
        x = self.pool2(x)
        x = self.dropout2(x)
        
        if layer_name == 'block2':
            return activations['block2']
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        activations['block3'] = x
        x = self.pool3(x)
        x = self.dropout3(x)
        
        if layer_name == 'block3':
            return activations['block3']
        
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        activations['block4'] = x
        
        if layer_name == 'block4':
            return activations['block4']
        
        raise ValueError(f"Unknown layer name: {layer_name}")


class SimplerBallCounterCNN(nn.Module):
    """
    A simpler CNN model for counting balls with fewer parameters
    and better interpretability
    """
    
    def __init__(self, num_classes=5):
        super(SimplerBallCounterCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x, return_features=False):
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        block1_features = x
        
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        block2_features = x
        
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        block3_features = x
        
        # Global average pooling
        x = self.gap(x)
        pooled_features = x
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layer
        x = self.fc(x)
        
        # If requested, return intermediate features for visualization
        if return_features:
            return {
                'block1': block1_features,
                'block2': block2_features,
                'block3': block3_features,
                'pooled': pooled_features,
                'output': x
            }
        
        return x
    
    def get_activation_maps(self, x, layer_name):
        """Get activation maps for visualization"""
        if layer_name == 'block1':
            return F.relu(self.bn1(self.conv1(x)))
        elif layer_name == 'block2':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            return F.relu(self.bn2(self.conv2(x)))
        elif layer_name == 'block3':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            return F.relu(self.bn3(self.conv3(x)))
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")
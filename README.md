# Ball Counter CNN

A PyTorch-based CNN solution for classifying images based on the number of balls present (1-5).

## Project Overview

This project implements a CNN model to count the number of balls in images. It includes:

- Data preprocessing with binary image conversion
- Two CNN architectures (simple and complex)
- Training and evaluation pipeline
- Visualization tools for model interpretability

## Project Structure

```
ball_counter/
│
├── data/
│   ├── raw/                   # Original images in folders 1, 2, 3, 4, 5
│   └── processed/             # Processed binary images
│
├── src/
│   ├── dataset.py             # Dataset classes and data loading utilities
│   ├── model.py               # CNN architecture definition
│   ├── train.py               # Training loop and logic
│   ├── visualization.py       # Visualization tools (Grad-CAM, filters, etc.)
│   └── utils.py               # Common utilities
│
├── tests/
│   ├── test_dataset.py        # Tests for dataset functionality
│   ├── test_model.py          # Tests for model architecture
│   └── test_integration.py    # End-to-end tests
│
├── notebooks/
│   ├── data_exploration.ipynb # Dataset analysis
│   └── model_evaluation.ipynb # Model analysis and visualization
│
├── main.py                    # Main script to run training and evaluation
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/ball-counter-cnn.git
cd ball-counter-cnn
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your images in the following structure:
```
data/raw/
  ├── 1/         # Images with 1 ball
  ├── 2/         # Images with 2 balls
  ├── 3/         # Images with 3 balls
  ├── 4/         # Images with 4 balls
  └── 5/         # Images with 5 balls
```

### Training

Run the training script:

```bash
python main.py --data_dir data/raw --samples 500 --batch_size 32 --epochs 30 --model_type simple --visualize
```

Arguments:
- `--data_dir`: Directory containing class folders
- `--samples`: Number of samples per class
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--model_type`: Model architecture to use (`simple` or `complex`)
- `--visualize`: Generate visualizations after training

### Testing

Run the tests:

```bash
pytest tests/
```

## Model Architecture

### Simple Model

A lightweight CNN with:
- 3 convolutional layers
- Global average pooling
- Fewer parameters for faster training and better interpretability

### Complex Model

A more sophisticated CNN with:
- 4 convolutional blocks (2 conv layers per block)
- More filters and features
- Better regularization with multiple dropout layers

## Visualization

The project provides several visualization tools:

1. **Grad-CAM**: Highlights areas of the image that influence the classification
2. **Filter Visualization**: Shows what patterns each filter detects
3. **Feature Maps**: Displays intermediate activations
4. **t-SNE**: Projects features to 2D space for cluster visualization
5. **Confusion Matrix**: Analyzes classification performance

## Results

The model achieves [your results here] accuracy on the test set. Visualization tools reveal that the model focuses on [your findings here].
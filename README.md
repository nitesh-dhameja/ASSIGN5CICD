# MNIST Classification with CI/CD Pipeline

## Project Overview
This project implements a lightweight Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model is designed to be efficient with less than 25,000 parameters while maintaining accuracy above 95%.

## Features
- Lightweight CNN architecture (13,242 parameters)
- Automated testing and validation
- CI/CD pipeline with GitHub Actions
- Progress tracking with tqdm
- Comprehensive model evaluation metrics

## Model Architecture
The model consists of:
- 2 Convolutional layers (4 and 8 filters)
- 2 MaxPooling layers
- 2 Fully connected layers (32 neurons and 10 outputs)
- ReLU activation functions

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- pytest
- tqdm

## Project Structure 
```
├── model/
│ ├── init.py
│ └── model.py # CNN model definition
├── tests/
│ └── test_model.py # Model tests
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # CI/CD configuration
├── train.py # Training script
├── requirements.txt # Project dependencies
├── .gitignore # Git ignore rules
└── README.md # Project documentation
  
```

## Getting Started

### Local Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python train.py
```

5. Run tests:
```bash
pytest tests/test_model.py -v
```

### GitHub Actions Pipeline
The CI/CD pipeline automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs validation tests

Tests verify that the model:
- Has less than 25,000 parameters
- Accepts 28x28 input images
- Outputs 10 classes
- Achieves >80% accuracy on test set

## Training Progress
The training script provides detailed progress information:
- Batch-wise progress bar
- Loss values
- Training and test accuracy every 100 batches
- Final model performance metrics

## Model Performance
- Parameters: ~13,242
- Expected accuracy: >80% on MNIST test set
- Training time: ~5 minutes on CPU

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
```

This README.md includes:
1. Clear project description
2. Detailed setup instructions
3. Project structure explanation
4. Model architecture details
5. Performance metrics
6. CI/CD pipeline information
7. Contributing guidelines
8. License information
9. Acknowledgments

The README provides both high-level overview and detailed technical information, making it easy for users to understand and get started with the project. The code structure section helps users navigate the repository, while the setup instructions provide step-by-step guidance for local development.
  
# Skin Disease Identifier

## Overview

**DermEnTai** is a deep learning-based image classification system designed to identify and differentiate between 23 types of skin diseases using a Convolutional Neural Network (CNN). With a dataset of 1380 labeled images, the model is trained to perform accurate and efficient multi-class classification.

This project implements an end-to-end pipeline including data preprocessing, CNN architecture for training, and a test script for evaluating model performance. It includes image enhancement techniques to improve feature clarity before feeding images into the neural network.

## Features

- **Image Preprocessing**: Enhances contrast and reduces noise using grayscale conversion, Gaussian blur, and histogram equalization.
- **Deep Learning Architecture**: Utilizes CNN layers with ReLU activation, batch normalization, max pooling, and dropout.
- **Multi-Class Classification**: Classifies images into 23 different skin disease categories.
- **Regularization Techniques**: Dropout layers reduce overfitting and improve generalization.
- **Evaluation Pipeline**: Separate script for testing and evaluating model performance on unseen data.

## Dataset

- **Images**: 1380 images
- **Classes**: 23 different skin diseases
- **Input Size**: 128x128 pixels, RGB

## Model Architecture

The CNN model is composed of the following layers:

1. **Input Layer**  
   - Input shape: `(128, 128, 3)` for RGB images

2. **Convolutional Layers**  
   - Multiple `Conv2D` layers with 32, 64, and 128 filters  
   - Kernel size: `2x2`  
   - Activation: ReLU  
   - Padding: 'same'

3. **Batch Normalization**  
   - Normalizes the outputs of convolutional layers  
   - Improves convergence speed and training stability

4. **MaxPooling2D**  
   - Pool size: `2x2`  
   - Stride: `2x2`  
   - Reduces spatial dimensions

5. **Dropout**  
   - Dropout rate: 0.25  
   - Helps prevent overfitting

6. **Flatten**  
   - Converts the 2D feature maps to 1D

7. **Dense Layer (Hidden)**  
   - 256 neurons  
   - Activation: ReLU

8. **Dropout Layer**  
   - Additional dropout for regularization

9. **Output Layer**  
   - 23 neurons (one per class)  
   - Activation: Softmax

## Image Enhancement

Before training, images are enhanced with the following steps:

1. **Convert to Grayscale**  
2. **Apply Gaussian Blur**  
3. **Histogram Equalization**

These enhancements help in improving contrast and reducing noise. Additional filters like median filtering or CLAHE can be experimented with for further improvements.

## File Structure

```bash
.
├── Dermentai-final.ipynb          # Main training script
├── test-dermentai-final.ipynb     # Script for testing the trained model
├── dataset/                       # Contains all images organized in subfolders by class
├── models/                        # Directory to save trained models
└── README.md                      # Project documentation
```

## Getting Started

### Prerequisites

- Python >= 3.7
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

Install the dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

### Running the Training Script

```bash
jupyter notebook Dermentai-final.ipynb
```

Follow the cells sequentially to preprocess images, define the CNN, and train the model.

### Running the Test Script

```bash
jupyter notebook test-dermentai-final.ipynb
```

Evaluates model performance on test images and visualizes predictions.

## Customization

- **Number of Filters / Layers**: Tune filter size and number of convolutional layers for deeper feature learning.
- **Image Size**: Adjust the input shape in the first layer for different image resolutions.
- **Dropout Rates**: Modify to control regularization intensity.
- **Additional Preprocessing**: Try advanced techniques like CLAHE or bilateral filtering.

## Results

The model demonstrates high performance in classifying the 23 categories of skin diseases. Accuracy and confusion matrix are visualized in the test script.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and contribute.

## Contact

For questions or suggestions, feel free to open an issue or pull request.

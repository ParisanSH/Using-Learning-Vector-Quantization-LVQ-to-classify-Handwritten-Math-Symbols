# Using Learning Vector Quantization(LVQ) to classify Handwritten Math Symbols

In this project, I have tried to train and test 1) a single-layer continuousneuron perceptron neural network and 2) a Learning Vector Quantization (LVQ) to
classify Handwritten Math Symbols.

Dataset:
➢ This dataset contains 1900 handwritten digits and arithmetic operators.
➢ Total number of classes: 19
➢ Most images resolution are 400 × 400 pixels and the others are 155 × 135.
➢ Each class contains 100 PNG images.

# Preprocessing 
steps which have been done:
- First, converting PNG images with 3 channels (RGB) to grayscale images with 256 intensities.
- Resizing images to 100 × 100.
- Selecting 80% of images of each class as train data and using the other 20% as test data.

# Part A: Continuous-neuron Perceptron
steps which have been done:
1) Training a single-layer continuous-neuron perceptron neural network.
2) Using test pictures to examine the generalization and reporting the Accuracy, Precision, and Recall.
3) Degrading the training images with 10% and 20% of noise (e.g. salt and pepper) to examine their robustness to noise. Using them as new test data and reporting the accuracy.

# Part B: Learning Vector Quantization (LVQ)
steps which have been done:
1) Training a Learning Vector Quantization (LVQ) neural network.
2) Repeating Part A steps and reporting the Accuracy, Precision, and Recall of test data and noisy train data and comparing them with the last Part.

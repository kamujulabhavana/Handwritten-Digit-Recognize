# Handwritten-Digit-Recognize
This project implements a Handwritten Digit Recognition System using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0–9). The images are preprocessed by normalizing pixel values and reshaping them into a format suitable for CNN input.

The CNN architecture consists of two convolutional layers with ReLU activation functions, each followed by max-pooling layers to extract important image features while reducing dimensionality. The extracted features are then flattened and passed through a fully connected dense layer before reaching the output layer, which uses the Softmax activation function to classify the input into one of the ten digit classes.

The model is compiled using the Adam optimizer and Sparse Categorical Crossentropy loss function and is trained for five epochs. After training, the model is evaluated on the test dataset to measure its classification accuracy. Finally, the system predicts the digit from a sample test image and displays both the predicted result and the actual handwritten image using Matplotlib.

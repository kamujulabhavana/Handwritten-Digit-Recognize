# Handwritten Digit Recognizer using CNN and MNIST Dataset

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load MNIST Dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# -----------------------------
# Normalize Data
# -----------------------------
X_train = X_train / 255.0
X_test = X_test / 255.0

# -----------------------------
# Reshape Data for CNN
# -----------------------------
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# -----------------------------
# Build CNN Model
# -----------------------------
model = models.Sequential()

# First Convolution Layer
model.add(layers.Conv2D(
    32,
    (3, 3),
    activation='relu',
    input_shape=(28, 28, 1)
))

# First Pooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolution Layer
model.add(layers.Conv2D(
    64,
    (3, 3),
    activation='relu'
))

# Second Pooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# Flatten Layer
model.add(layers.Flatten())

# Dense Hidden Layer
model.add(layers.Dense(64, activation='relu'))

# Output Layer (0-9 digits)
model.add(layers.Dense(10, activation='softmax'))

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# -----------------------------
# Evaluate Model
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_acc)

# -----------------------------
# Make Predictions
# -----------------------------
predictions = model.predict(X_test)

# Predict first image
predicted_digit = np.argmax(predictions[0])

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[0])

# -----------------------------
# Display Image
# -----------------------------
plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.axis('off')
plt.show()

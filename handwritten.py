from IPython import get_ipython
from IPython.display import display
# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to include the channel dimension
x_train = np.expand_dims(x_train, -1)  # shape becomes (num_samples, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# Predict the classes for the test set
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate and print accuracy for each digit
print("\nAccuracy for each digit:")
for i in range(num_classes):
    # Get indices of all test samples with actual label i
    indices = np.where(y_true == i)[0]
    if len(indices) > 0:
        # Get the predicted labels for these samples
        predicted_labels = y_pred[indices]
        # Calculate the number of correct predictions
        correct_predictions = np.sum(predicted_labels == i)
        # Calculate accuracy for digit i
        accuracy_i = correct_predictions / len(indices)
        print(f"Digit {i}: {accuracy_i:.4f}")
    else:
        print(f"Digit {i}: No test samples for this digit.")


# Display sample predictions
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    pred_label = np.argmax(predictions[i])
    actual_label = np.argmax(y_test[i])
    confidence = np.max(predictions[i])
    plt.title(f"Prediction: {pred_label} ({confidence:.2f}), Actual: {actual_label}")
    plt.axis('off')
    plt.show()

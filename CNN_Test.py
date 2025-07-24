import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import random

# Load and Normalize the Dataset

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Scale pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add a channel dimension for CNN input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# Define the CNN Architecture


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# Compile the Model

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Train the Model


history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_test, y_test),
    verbose=1
)


# Plot Training Performance


# Accuracy over epochs
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Loss over epochs
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Predict and Display Results


predictions = model.predict(X_test)

# Show 5 random test images and predictions
random_indices = random.sample(range(len(X_test)), 5)

plt.figure(figsize=(10, 4))
for i, idx in enumerate(random_indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    predicted_label = np.argmax(predictions[idx])
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

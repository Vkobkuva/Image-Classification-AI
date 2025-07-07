"""
Image Classification AI - Portfolio Project
Author: [Your Name]
Description: Train a CNN to classify images from the CIFAR-10 dataset using TensorFlow.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and preprocess CIFAR-10 dataset
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    return train_images, train_labels, test_images, test_labels

# 2. Visualize some sample images
def plot_sample_images(images, labels, class_names):
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i][0]])
    plt.suptitle('Sample Images from CIFAR-10')
    plt.show()

# 3. Build the CNN model
def build_model():
    cnn = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return cnn

# 4. Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 5. Train and evaluate the model
def train_and_evaluate():
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    X_train, y_train, X_test, y_test = load_data()
    plot_sample_images(X_train, y_train, class_names)
    model = build_model()
    print("Starting training...")
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    plot_history(history)
    print("Evaluating on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.2f}")
    # Save the trained model
    model.save('cifar10_cnn_model.h5')
    print("Model saved as 'cifar10_cnn_model.h5'")
    # Example prediction
    idx = np.random.randint(0, X_test.shape[0])
    sample = np.expand_dims(X_test[idx], axis=0)
    prediction = np.argmax(model.predict(sample), axis=1)[0]
    plt.imshow(X_test[idx])
    plt.title(f"Predicted: {class_names[prediction]}, True: {class_names[y_test[idx][0]]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()

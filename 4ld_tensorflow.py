import pandas as pd
import numpy as np
import keras
from keras import layers
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# """
# Function for plotting and saving original vs augmented images
# """
def original_augmented(original, augmented, labels):
    # Plot original and augmented images together
    plt.figure(figsize=(10, 10))
    # Plot original images
    for i in range(5):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')  # Original images are grayscale, so we use cmap='gray'
        plt.title("Original: {}".format(np.argmax(labels[i])))  # Display original label
        plt.axis("off")
    # Plot augmented images
        ax = plt.subplot(2, 5, i + 6)  # Start plotting on the second row
        plt.imshow(augmented[i].numpy().squeeze(), cmap='gray')  # Augmented images are still in tensor format, so we use .numpy()
        plt.title("Augmented: {}".format(np.argmax(labels[i])))  # Display original label
        plt.axis("off")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("tensorflow_files\original vs augmented images.png")
    plt.show()

# """
# ## Function that plots 5 predicted images of the best and last models
# """
def best_last(x_test_data, best_model_accuracy, last_model_accuracy, best_model, last_model):
    # Predict probabilities for the test set
    probabilities_best = best_model.predict(x_test_data)
    probabilities_last = last_model.predict(x_test_data)

    # Get predicted labels and guessing percentages from the best model
    predicted_labels_best = np.argmax(probabilities_best, axis=1)
    guessing_percentages_best = np.max(probabilities_best, axis=1)

    # Get predicted labels and guessing percentages from the last model
    predicted_labels_last = np.argmax(probabilities_last, axis=1)
    guessing_percentages_last = np.max(probabilities_last, axis=1)

    # Plot best models predictions
    plt.figure(figsize=(14, 8))
    for i in range (5):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(x_test_data[i].squeeze(), cmap='gray')
        plt.title("Predicted: {} ({:.2f}%)".format(predicted_labels_best[i], guessing_percentages_best[i] * 100))
        plt.axis('off')
    # Plot last models predictions
        ax = plt.subplot(2, 5, i + 6)
        plt.imshow(x_test_data[i].squeeze(), cmap='gray')
        plt.title("Predicted: {} ({:.2f}%)".format(predicted_labels_last[i], guessing_percentages_last[i] * 100))
        plt.axis('off')

    plt.text(0.5, 0.97, "Best Model Accuracy: {:.2f}%".format(best_model_accuracy * 100), horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure, fontsize=12)
    plt.text(0.5, 0.47, "Last Model Accuracy: {:.2f}%".format(last_model_accuracy * 100), horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure, fontsize=12)


    plt.tight_layout()
    plt.savefig("last vs best model.png")
    plt.show()


# """
# ## Preparing the data
# """
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# """
# ## Data augmentation
# """
# Data augmentation layers
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),  # Horizontal flipping
    layers.RandomFlip("vertical"),    # Vertical flipping, i did this here BUT I SHOULDNT HAVE, data augmentation shouldnt change the nature of data (f.e. if 9 is flipped it becomes 6, thats bad)
    layers.RandomRotation(0.1),       # Random rotation by up to 10 degrees
    layers.RandomZoom(0.2),           # Random zoom by up to 20%
    layers.RandomTranslation(height_factor=0.2, width_factor=0.2),  # Random translation
    layers.RandomContrast(factor=0.2)  # Random contrast adjustment
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

x_train_aug = data_augmentation(x_train)

# """
# ## Plotting original vs augmented images
# """
original_augmented(x_train,x_train_aug,y_train)


# """
# ## Build the model
# """
# # Define the model architecture with modified conditions
model = keras.Sequential(
    [
        # Input layer with specified input shape
        keras.Input(shape=input_shape),
        # Convolutional layer with 32 filters and relu activation
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),  # Modify padding parameter
        # MaxPooling layer with pool size (2, 2)
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Additional Convolutional layer with 32 filters and relu activation
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),  # Modify padding parameter
        # Additional Convolutional layer with 64 filters and relu activation
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),  # Modify padding parameter
        # MaxPooling layer with pool size (2, 2)
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Dropout layer with dropout rate of 0.5
        layers.Dropout(0.5),
        # Flatten layer to flatten the input
        layers.Flatten(), # Transforms matrix data to a vector
        # Dense layer with 128 units and relu activation
        layers.Dense(128, activation="relu"),
        # Additional Dense layer with 64 units and relu activation
        layers.Dense(64, activation="relu"),
        # Output layer with num_classes units and softmax activation for classification
        layers.Dense(num_classes, activation="softmax")
    ]
)
model.summary()

# """
# ## Training the model and saving models with lowest validation loss and the latest model
# """
batch_size = 128
epochs = 3

# Define a callback to save the model with the lowest validation loss
checkpoint_callback = keras.callbacks.ModelCheckpoint("tensorflow_files\\best_model.keras", save_best_only=True, monitor='val_loss', mode='min', verbose = 1) # verbose lets you see in console wich model performed better than the last one 
# and whether it was chnged to best_model.keras

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model with the checkpoint callback
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback])

# Save the model from the last epoch
model.save("tensorflow_files\last_epoch_model.keras")


"""
## Evaluate the trained model
"""
# Load the best model with the lowest validation loss
best_model = keras.models.load_model("tensorflow_files\\best_model.keras")
last_model = keras.models.load_model("tensorflow_files\\last_epoch_model.keras")

# Evaluate the best model on the test set
test_loss, test_accuracy_best = best_model.evaluate(x_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy_best)

test_loss, test_accuracy_last = last_model.evaluate(x_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy_last)


"""
## Ploting prediction results of best and last models
"""
best_last(x_test, test_accuracy_best, test_accuracy_last, best_model, last_model)

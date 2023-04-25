import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

# Set the desired size of the input images
IMAGE_SIZE = (128, 128)

# Set the number of classes in the dataset
NUM_CLASSES = 11

# Set the path to the dataset
path = 'raw-img'
path_test = 'raw-img\1\farfalla0.jpg'
#****____LABELS____****#
# 1 = butterfly
# 2 = cat
# 3 = chicken
# 4 = cow
# 5 = dog
# 6 = elephant
# 7 = horse
# 8 = sheep
# 9 = spider
# 10 = squirrel

def load_dataset(path):
    data = []
    labels = []
    # Loop over the directories in the dataset
    for animal in os.listdir(path):
        # Get the path to the animal directory
        animal_path = os.path.join(path, animal)
        # Loop over the images in the animal directory
        for image_file in os.listdir(animal_path):
            # Get the path to the image file
            image_path = os.path.join(animal_path, image_file)
            # Load the image and resize it to the desired size
            src = cv2.imread(image_path, 0)
            image = cv2.resize(src, IMAGE_SIZE)
            cv2.imwrite(image_path,image)
            # Append the image data and label to the lists
            data.append(image)
            labels.append(animal)
    # Convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def preprocess_dataset(data, labels):
    # Convert the labels to one-hot encoding
    labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    # Normalize the image data
    data = data / 255.0
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train = np.reshape(X_train, (-1, 128, 128, 1))
    X_test = np.reshape(X_test, (-1, 128, 128, 1))
    return X_train, X_test, y_train, y_test

def build_model():
    model = Sequential()
    # Add the first convolutional layer with 32 filters and a 3x3 kernel
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)))
    # Add a max pooling layer with a 2x2 pool size
    model.add(MaxPooling2D((2, 2)))
    # Add a second convolutional layer with 64 filters and a 3x3 kernel
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Add another max pooling layer with a 2x2 pool size
    model.add(MaxPooling2D((2, 2)))
    # Add a third convolutional layer with 128 filters and a 3x3 kernel
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # Add another max pooling layer with a 2x2 pool size
    model.add(MaxPooling2D((2, 2)))
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    # Add a fully connected layer with 512 units and a dropout rate of 0.5
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # Add the output layer with NUM_CLASSES units and a softmax activation
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def train_model(X_train, X_test, y_train, y_test):
# Build the model
    model = build_model()
# Compile the model with categorical crossentropy loss and Adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Set up a checkpoint to save the model with the highest validation accuracy
    checkpoint_path = 'animal_classifier.h5'
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=False, monitor='val_accuracy', mode='max')
# Train the model with batch size of 32 and 50 epochs
    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])
# Load the saved model with the highest validation accuracy
    model = tf.keras.models.load_model(checkpoint_path)
    return model

data, labels = load_dataset(path)
X_train, X_test, y_train, y_test = preprocess_dataset(data, labels)
model = train_model(X_train, X_test, y_train, y_test)
model.save('animal_classifier.h5')
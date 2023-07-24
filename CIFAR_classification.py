# Importing the libraries
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

#loading and splitting the CIFAR10 dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# The data specifics - 
print("The shape of the train dataset is", x_train.shape)
print("The shape of the test dataset is", x_test.shape)

print("The length of the train dataset is", len(x_train))
print("The length of the test dataset is", len(x_test))

# The network architecture 

model = keras.Sequential([layers.Dense(512, activation="relu"), layers.Dense(10, activation="softmax")])

#preparing the data
train_images = x_train.reshape((50000, 32, 32, 3))
train_images = x_train.astype("float32") / 255
test_images = x_test.reshape((10000,32,32, 3))
test_images = x_test.astype("float32") / 255

# training the model on our dataset and validating it with test set to check for overfitting or under
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compiling the model
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# fitting the model
model.fit(train_images, y_train, epochs=10, batch_size=128)

model.summary()

# Testing the accuracy
test_loss, test_acc = model.evaluate(test_images, y_test)
print(f"The Test accuracy: {test_acc:.3f}")


# ADDING ANOTHER LAYER TO CHECK IF IT INCREASES ACCURACY
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, y_train, epochs=10)

model.summary()

#Testing the models accuray after adding an additional layer
test_loss, test_acc = model.evaluate(test_images, y_test)
print(f"The Test accuracy: {test_acc:.3f}")
# Hello World Tensorflow

# Import libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define the model
model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

# Define train data
x_train = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype = "float")
y_train = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = "float")

# Compile the model
model.compile(loss = "mean_squared_error", optimizer = "sgd")

# Fit the model
model.fit(x_train, y_train, epochs = 500)

# Test the model
x_test = [10.0]
y_test = model.predict(x_test)[0]
y = [19.0]
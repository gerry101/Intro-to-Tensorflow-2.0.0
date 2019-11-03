# NN - Converting Celcius to Farenheit

# Import libraries
import logging
import tensorflow as tf
print(f"\nUsing tensorflow v.{tf.__version__}")
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import numpy as np

# Epoch on_end callback
class endEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get("accuracy") > 0.95):
            epoch_acc = logs.get("accuracy")
            print(f"Accuracy is currently {epoch_acc}. Cancelling training")
            self.model.stop_training = True
    
callbacks = endEpochCallback()

# Get training data
x_train = np.array([-40.0, -10.0,  0.0,  8.0, 15.0, 22.0,  38.0], dtype = "float")
y_train = np.array([-40.0,  14.0, 32.0, 46.0, 59.0, 72.0, 100.0], dtype = "float")

# Define the model
model = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])

# Compile the model
model.compile(loss = "mean_squared_error", optimizer = tf.optimizers.Adam(0.1), metrics = ["accuracy"])

# fit the model
model.fit(x_train, y_train, epochs = 1500, callbacks = [callbacks])

# Predict value using model
x_test = 5.0
y_actual = 41.0
y_test = model.predict([5.0])[0]

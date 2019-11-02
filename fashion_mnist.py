# Fashion MNIST Classtfication

# Import libraries
import tensorflow as tf
print(f"Using tensorflow v.{tf.__version__}")

# Define epoch accuracy callback
class epochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get("accuracy") > 0.95):
            epoch_acc = logs.get("accuracy")
            print(f"\n\nCurrent epoch accuracy is {epoch_acc}. Cancelling training.\n")
            self.model.stop_training = True

callbacks = epochCallback()

# Get fashion_mnist data
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize image data
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define the model
model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation = tf.nn.relu),
            tf.keras.layers.Dense(10, activation = tf.nn.softmax)
        ])

# Compile the model
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Fit the model
model.fit(training_images, training_labels, epochs = 5, callbacks = [callbacks])

# Evaluate the model
model.evaluate(test_images, test_labels)

7
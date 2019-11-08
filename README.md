# Intro-to-Tensorflow-2.0.0
Introductory Projects to Tensorflow 2.x and Data Preprocessing in Python.

### Projects
Projects presented in increasing complexity
  - [Hello Tensorflow](https://github.com/gerry101/Intro-to-Tensorflow-2.0.0/blob/master/hello_tensorflow.py): Simple single-layer neural network that predicts the y value of a linear equation (y = 2x - 1).
    - Single dense layer with single neuron.
    - Makes use of the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) loss function and the [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) optimizer.
  - [Celcius to Farenheit](https://github.com/gerry101/Intro-to-Tensorflow-2.0.0/blob/master/celcius_to_farenheit.py): Simple neural network that is able to convert celcius degrees to Farenheit. (Not that this can't be achieved by pen and paper but where's the fun in that!? :) ).
    - Single dense layer with single neuron.
    - Makes use of the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) loss function and the [adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) optimizer.
    - Uses early stopping by implementing Tensorflow's ```on_epoch_end()``` function so as to prevent overfitting.
  - [Fashion MNIST](https://github.com/gerry101/Intro-to-Tensorflow-2.0.0/blob/master/fashion_mnist.py): A simple Convolutional Neural Network that predicts the type of clothing artifact from a dataset of 10 different classes.
    - Implements Keras' ```Conv2D``` and ```MaxPooling2D``` layers so as increase feature detection accuracy by the model while decreasing input to be processed by the model's dense layers.
    - Uses early stopping by implementing Tensorflow's ```on_epoch_end()``` function so as to prevent overfitting.
    - Uses Keras' ```sparse_categorical_crossentropy``` loss function to handle multi-class classification.
  - [Cats vs Dogs](https://github.com/gerry101/Intro-to-Tensorflow-2.0.0/blob/master/Cats_vs_Dogs.ipynb): CNN trained on over 2,000 images to predict whether image contains a cat or a dog.
    - Uses early stopping by implementing Tensorflow's ```on_epoch_end()``` function so as to prevent overfitting.
    - Makes use of Tensorflow's ```ImageDataGenerator``` so as to simplify image data preprocessing.
    - Implements image augmentation on the test dataset so as ti increase the variation of image styles that the model is trained on.
    - Implements the Keras ```Dropout``` layer so as to prevent overfitting while potentially increasing both training and validation accuracies.
    - Uses the ```RMSprop``` optimizer so as to have maximum control over the model's learning rate.
    # 
  
  

###### Happy Hacking,
Gerry.

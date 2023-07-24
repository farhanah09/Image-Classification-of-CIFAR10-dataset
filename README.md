
# Introduction

We'll cover the following steps:

- Setup
- Dataset
- Preprocessing
- Constructing the Network Architecture
- Training the CNN Model
- Evaluating Model Accuracy

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is divided into 50,000 training images and 10,000 testing images. You can download the dataset using TensorFlow's built-in datasets or from external sources.

Before feeding the images into the CNN, some preprocessing steps are necessary. Typical preprocessing steps include:

Resizing the images to a fixed size (e.g., 32x32) to match the input size of the CNN.
Normalizing pixel values to the range [0, 1].
One-hot encoding the class labels (if using categorical cross-entropy loss).

Build the CNN model using TensorFlow's Keras API. You can define the model architecture by stacking convolutional layers, pooling layers, and fully connected (dense) layers. The number of layers, filter sizes, and the number of units in the dense layers can be adjusted based on the complexity of the problem.

Train the constructed CNN model using the training dataset. Adjust the batch size, number of epochs, and other hyperparameters based on your computational resources and convergence performance. After training, evaluate the model's accuracy on the test dataset to assess its performance.
## Requirements

#### Install all the required libraries for the code

```http
  pip install requirements.txt
```

## Acknowledgements

This code is primarily adapted from Chapters 2 and 8 of the book [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff).

This code was a part of the Big Data (Course No. 45980) for Spring 2023 at Tepper School, Carnegie Mellon University 

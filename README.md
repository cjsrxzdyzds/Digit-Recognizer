# Digit-Reconizer
This code implements a Convolutional Neural Network (CNN) model for digit recognition using the MNIST dataset. The model architecture consists of multiple convolutional layers, batch normalization, activation functions, max pooling, and dropout layers. The code utilizes data augmentation techniques, learning rate scheduling, and early stopping callbacks to improve model performance. The trained model is then used to make predictions on the test set, and the results are saved to a CSV file for submission.

Digit Recognizer CNN Model\
This repository contains code for training a Convolutional Neural Network (CNN) model to recognize handwritten digits using the MNIST dataset. The model is implemented using TensorFlow and Keras.

Dataset\
The MNIST dataset is used for training and testing the model. It consists of 70,000 grayscale images of handwritten digits, with 60,000 images for training and 10,000 images for testing. The dataset is loaded from the Kaggle competition "Digit Recognizer".

Model Architecture\
The CNN model architecture consists of the following layers:

Convolutional layers with increasing number of filters (32, 64, 128, 256)\
Batch normalization layers\
ReLU activation functions\
Max pooling layers\
Dropout layers\
Flatten layer\
Dense layers with regularization\
Softmax activation for output\
Training


The model is trained using the following techniques:

Data augmentation using rotation, zoom, width shift, and height shift\
Learning rate scheduling with a decreasing learning rate over epochs\
Early stopping based on validation loss\
RMSprop optimizer with a learning rate of 0.001\
Categorical cross-entropy loss\
Accuracy metric\
Prediction

After training, the model is used to make predictions on the test set. The predicted labels are saved to a CSV file named "submission.csv" for submission to the Kaggle competition.

Requirements\
Python 3.x\
TensorFlow 2.x\
Keras\
NumPy\
Pandas\
scikit-learn

Usage\
Download the MNIST dataset from the Kaggle competition "Digit Recognizer".\
Update the file paths in the code to point to the downloaded dataset.\
Run the code in a Jupyter Notebook or Python environment.\
The trained model will be saved as "mnist_cnn_improved_gpu.h5".\
The predictions will be saved in "submission.csv".

License\
This project is licensed under the MIT License.

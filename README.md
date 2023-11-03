# Sarcasm Detector LSTM - RNN Model

This GitHub repository contains a project for sarcasm detection using LSTM (Long Short-Term Memory) neural networks. The project leverages natural language processing and deep learning techniques to identify whether a given text is sarcastic or not. 

## Overview

Sarcasm detection is a challenging task in natural language processing, as it often involves subtle cues and context understanding. In this project, we build a deep learning model that can classify text as sarcastic or non-sarcastic based on a labeled dataset.

The project includes the following key components:

1. **Data Preparation**: We start by importing a labeled dataset of text and their corresponding labels, which indicate whether the text is sarcastic or not. We preprocess the text data by lowercasing, removing special characters, and tokenizing the text.

2. **Data Visualization**: We visualize the distribution of sarcastic and non-sarcastic examples in the dataset using a countplot to understand the balance of the classes.

3. **Text Tokenization**: We tokenize the text data and perform padding to ensure that all sequences have the same length. This processed data is used as input to our deep learning model.

4. **Model Architecture**: We design a sequential deep learning model using Keras, which includes an Embedding layer, SpatialDropout1D layer, LSTM layer, and a Dense output layer with softmax activation. This model is trained to classify text as either sarcastic or non-sarcastic.

5. **Model Training**: The model is trained on the training dataset with 25 epochs, and we monitor loss and accuracy during training. The training process aims to optimize the model's ability to classify text.

6. **Model Evaluation**: We evaluate the model's performance on a validation dataset and calculate accuracy and loss. We also analyze the accuracy of the model in distinguishing sarcasm and non-sarcasm.

7. **Inference**: The trained model can be used for inference. You can input a text sentence, and the model will predict whether it is sarcastic or not.

8. **Model Saving**: The trained model is serialized to JSON and saved to an HDF5 file for future use.

## Usage

To run the project or use the trained model:

1. Clone the GitHub repository.
2. Install the required libraries mentioned in the code.
3. Use Jupyter Notebook or any Python environment to run the provided Jupyter notebook containing the code.

## Results

The project aims to achieve accurate classification of text as sarcastic or non-sarcastic. The model's performance can be assessed using accuracy, precision, recall, and F1-score metrics.

## Dataset

The project uses a labeled dataset containing examples of sarcastic and non-sarcastic text. The dataset is loaded from a CSV file and is expected to have a "text" column for the text data and a "humor" column for the labels.

## Dependencies

- Python 3
- TensorFlow/Keras
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-Learn


Feel free to modify and use this code for your sarcasm detection projects. If you find this project useful or have any feedback, please don't hesitate to get in touch or create an issue.

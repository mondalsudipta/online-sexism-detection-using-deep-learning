# Detection of Online Sexism

This repository contains the implementation of an explainable system for detecting online sexism. The task is defined into two subtasks: Task A for binary sexism detection and Task B for categorizing sexist content.

## Introduction

Online sexism is a pervasive issue that poses significant challenges to creating inclusive and safe online spaces. This project aims to mitigate the harm caused by online sexism by developing an automated detection system, focusing on explainable detection to make online spaces more accessible and welcoming for everyone.

## Dataset

We utilized a dataset from online platforms, containing posts labeled with their corresponding sexism status and category.

### Data Cleaning

We performed several preprocessing steps on the text data before training the models:

- Removing non-alphanumeric characters except whitespace.
- Converting text to lowercase.
- Removing extra whitespaces.
- Removing stop words using the NLTK library.
- Lemmatizing words to reduce them to their base form.

### Dataset Structure

The dataset comprises five columns. For our tasks:

- **Task A:** Relevant columns are 'text' and 'label_sexist'.
- **Task B:** Relevant columns are 'text' and 'label_category'.

We created two subsets of the dataframe:

- **df_A:** Contains 'text' and 'label_sexist' columns.
- **df_B:** Contains 'text' and 'label_category' columns.

### Label Encoding

- **Task A:** Label encoding was performed with 'not sexist' as 0 and 'sexist' as 1.
- **Task B:** One-hot encoding was used for the four class labels.

### Data Splitting

The data was split into training and testing sets using an 80-20 ratio, with tokenized texts as features and encoded columns as targets.

## Model

### Neural Network Model Architecture

We built a neural network model using Keras for both tasks. The architecture is as follows:

- **Input layer:** Dense layer with 256 units and ReLU activation function.
- **Dropout layer:** Dropout rate of 0.65 to prevent overfitting.
- **Output layer for Task A:** Dense layer with 1 unit and sigmoid activation function.
- **Output layer for Task B:** Dense layer with 4 units and softmax activation function.

### Additional Techniques

We experimented with various hyperparameters, such as the number of units in the hidden layers, dropout rates, and activation functions, to optimize model performance. Techniques like learning rate scheduling and data augmentation were also employed to enhance the model's generalization capability.

## Results

### Task A: Binary Sexism Detection

- **Epochs:** 60
- **Batch Size:** 40
- **Training Accuracy:** 98.75%
- **Testing Accuracy:** 79.39%
- **F1 Score:** 77.06%

### Task B: Categorizing Sexist Content

- **Epochs:** 60
- **Batch Size:** 40
- **Training Accuracy:** 94.85%
- **Testing Accuracy:** 45.59%
- **F1 Score:** 43.97%

## Technology Used

- **Data Analysis and Manipulation:**
  - `pandas`
  - `numpy`
  - `re`

- **Text Processing:**
  - `nltk` (for stopwords, lemmatization, tokenization)
  - `gensim` (for Word2Vec)

- **Machine Learning:**
  - `scikit-learn` (for train_test_split, TfidfVectorizer, LabelEncoder)
  - `imblearn` (for SMOTE and RandomUnderSampler)

- **Deep Learning:**
  - `keras` (for building and training neural network models)
  - `tensorflow` (as the backend for Keras)

- **Visualization:**
  - `matplotlib`

# EEG Seizure Detection Project

## Overview
This project focuses on detecting epileptic seizures in EEG (Electroencephalogram) signals using machine learning and image processing techniques. The project involves data preprocessing, feature extraction, model training, and evaluation.

## Contents
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Image Processing](#image-processing)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Data
The EEG data is stored in a CSV file (`eeg_data.csv`) and includes signal values and labels indicating whether a seizure occurred.

## Data Preprocessing
- The CSV file is loaded into a Pandas DataFrame.
- Missing values are handled by dropping rows with missing features.
- Feature extraction is performed, and the data is split into training and testing sets.
- Data normalization is applied using StandardScaler.

## Model Training
- Support Vector Machine (SVM) classifier is used for training.
- The model is evaluated using accuracy, precision, recall, and F1-score.

## Model Evaluation
- Confusion matrix is computed to assess the performance of the trained model.
- Precision, recall, and F1-score are calculated with average='weighted'.
- **Accuracy: 0.57** *(The actual accuracy value)*
- Precision: 0.61
- Recall: 0.57
- F1-score: 0.55

## Image Processing
- An image of EEG signals is loaded and preprocessed.
- The pre-trained VGG16 model is used to extract features from the image.
- Trained non-epileptic and epileptic models are loaded.
- Predictions are made using both VGG16 and the loaded models.
- The final prediction is determined based on the comparison of probabilities.

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Follow the instructions in the project code for data loading, preprocessing, model training, and image processing.

## Dependencies
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- numpy
- tensorflow
- skimage
- keras

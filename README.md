# epilepsy_project
This project focuses on developing advanced AI/ML algorithms for accurate and efficient detection of epilepsy signals, enabling early diagnosis &amp; personalized treatment.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage import io, transform
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# Load the data into a Pandas DataFrame
data = pd.read_csv('eeg_data.csv')

# Data Exploration - Summary Statistics
summary_stats = data.describe()
print(summary_stats)

# Data Exploration - Visualization
# Histograms for numerical features (signals)
data.iloc[:, 1:-1].hist(figsize=(15, 30), bins=20)  # Excluding the 'Signals' and 'y' columns
plt.show()

# Box plots for numerical features (signals)
data.iloc[:, 1:-1].boxplot(figsize=(15, 6))  # Excluding the 'Signals' and 'y' columns
plt.show()

# Data Exploration - Handling Missing Values
missing_values = data.isnull().sum()
print(missing_values)

# Data Exploration - Data Profiling
data_profile = data.info()
print(data_profile)

# Data Preprocessing - Handling Missing Values
data.dropna(inplace=True)

# Data Preprocessing - Removing Duplicates
data.drop_duplicates(inplace=True)

# Extract Features and Target Variable
X = data.iloc[:, 1:-1]  # Extract all columns except 'Signals' and 'y' as features
y = data['y']  # Extract the 'y' column as the target variable

# Separate the 'Signals' column to use it as an index
data.set_index('Signals', inplace=True)

# Transpose the DataFrame so that signals are in rows and data values are in columns
data = data.T
# Load the Excel sheet
data = pd.read_csv('eeg_data.csv')

# Separate the non-epileptic and epileptic signals based on the 'y' column
non_epileptic_data = data[data['y'] != 1]
epileptic_data = data[data['y'] == 1]

# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# Plot non-epileptic signals
for i in range(len(non_epileptic_data)):
    signal_values = non_epileptic_data.iloc[i, 1:-1]
    ax1.plot(signal_values)

# Set labels and title for the non-epileptic plot
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.set_title('Non-Epileptic Signals')

# Plot epileptic signals
for i in range(len(epileptic_data)):
    signal_values = epileptic_data.iloc[i, 1:-1]
    ax2.plot(signal_values)

# Set labels and title for the epileptic plot
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitude')
ax2.set_title('Epileptic Signals')

# Adjust the layout
plt.tight_layout()

# Show the plots
plt.show()
# Load the data into a Pandas DataFrame
data = pd.read_csv('eeg_data.csv')

# Extract Features and Target Variable
X = data.iloc[:, 1:-1]  # Extract all columns except 'Signals' and 'y' as features
y = data['y']  # Extract the 'y' column as the target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate non-epileptic and epileptic signals based on the target variable
non_epileptic_train = X_train[y_train != 1]
non_epileptic_test = X_test[y_test != 1]

epileptic_train = X_train[y_train == 1]
epileptic_test = X_test[y_test == 1]

# Perform feature selection
k = 50  # Number of top features to select
selector = SelectKBest(score_func=lambda X, y: chi2(X, y), k=k)

# Train the model for non-epileptic signals
non_epileptic_selector = selector.fit(non_epileptic_train.abs(), y_train[y_train != 1])
non_epileptic_train_selected = non_epileptic_selector.transform(non_epileptic_train)
non_epileptic_test_selected = non_epileptic_selector.transform(non_epileptic_test)

non_epileptic_model = RandomForestClassifier(n_estimators=100, random_state=42)
non_epileptic_model.fit(non_epileptic_train_selected, y_train[y_train != 1])

# Train the model for epileptic signals
epileptic_selector = selector.fit(epileptic_train.abs(), y_train[y_train == 1])
epileptic_train_selected = epileptic_selector.transform(epileptic_train)
epileptic_test_selected = epileptic_selector.transform(epileptic_test)

epileptic_model = RandomForestClassifier(n_estimators=100, random_state=42)
epileptic_model.fit(epileptic_train_selected, y_train[y_train == 1])

# Make predictions on the test set for both models
non_epileptic_pred = non_epileptic_model.predict(non_epileptic_test_selected)
epileptic_pred = epileptic_model.predict(epileptic_test_selected)

# Evaluate the models
non_epileptic_accuracy = accuracy_score(y_test[y_test != 1], non_epileptic_pred)
epileptic_accuracy = accuracy_score(y_test[y_test == 1], epileptic_pred)

non_epileptic_conf_matrix = confusion_matrix(y_test[y_test != 1], non_epileptic_pred)
epileptic_conf_matrix = confusion_matrix(y_test[y_test == 1], epileptic_pred)

non_epileptic_report = classification_report(y_test[y_test != 1], non_epileptic_pred)
epileptic_report = classification_report(y_test[y_test == 1], epileptic_pred)

# Print the evaluation results for both models
print("Non-Epileptic Model Accuracy:", non_epileptic_accuracy)
print("Non-Epileptic Model Confusion Matrix:\n", non_epileptic_conf_matrix)
print("Non-Epileptic Model Classification Report:\n", non_epileptic_report)

print("Epileptic Model Accuracy:", epileptic_accuracy)
print("Epileptic Model Confusion Matrix:\n", epileptic_conf_matrix)
print("Epileptic Model Classification Report:\n", epileptic_report)
# Save the models to files
joblib.dump(non_epileptic_model, 'non_epileptic_model.joblib')
joblib.dump(epileptic_model, 'epileptic_model.joblib')

# Train the feature selector
k = 50  # Number of top features to select
def custom_chi2(X, y):
    return chi2(X, y)[0]

selector = SelectKBest(score_func=custom_chi2, k=k)

# Save the trained feature selector model
joblib.dump(selector, 'feature_selector.joblib')
print("Feature selector model saved successfully!")

print("Models saved successfully!")
# Load the models from files
loaded_non_epileptic_model = joblib.load('non_epileptic_model.joblib')
loaded_epileptic_model = joblib.load('epileptic_model.joblib')

# Make predictions using the loaded models
non_epileptic_pred = loaded_non_epileptic_model.predict(non_epileptic_test_selected)
epileptic_pred = loaded_epileptic_model.predict(epileptic_test_selected)

# Load the trained feature selector model
feature_selector = joblib.load('feature_selector.joblib')
print("Feature selector model loaded successfully!")
training_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Convert the training data to a pandas DataFrame
df = pd.DataFrame(training_data)

# Define and fit the MinMaxScaler on training data
scaler = MinMaxScaler()
scaler.fit(df)

# Save the scaler to a file
scaler_filename = 'scaler.joblib'
joblib.dump(scaler, scaler_filename)
# Load and preprocess the image
image_path = 'new_data.jpg'
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = tf.expand_dims(img_array, axis=0)

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load trained models
non_epileptic_model = joblib.load('non_epileptic_model.joblib')
epileptic_model = joblib.load('epileptic_model.joblib')

# Make predictions on the image
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=3)[0]
print('Top predictions:')
for pred in decoded_predictions:
    print(f'{pred[1]}: {pred[2]*100:.2f}%')

# Make predictions on the image
prediction = model.predict(img_array)
predicted_class = 'Epilepsy' if prediction[0][0] > 0.5 else 'Non-Epilepsy'

print('Predicted Class:', predicted_class)

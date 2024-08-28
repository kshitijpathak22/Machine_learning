# **Early Detection of Depression and Anxiety**

**Description**
This project leverages machine learning to identify early signs of depression and anxiety based on health data, including BMI, age, and gender. By applying Support Vector Machines (SVM) and Neural Networks, we aim to predict psychological conditions effectively, aiding in early intervention strategies.

**Features**
Data cleaning and preprocessing
Correlation analysis between different variables
Deployment of machine learning models for predictive analytics:
Neural Network models for classification tasks
SVM for comparative analysis
Application of SMOTE for addressing class imbalance in training datasets
Results visualization through correlation matrices and confusion matrices

**Prerequisites**
Required libraries:
numpy
pandas
tensorflow
scikit-learn
imblearn
seaborn
matplotlib
Installing

**Instructions for setting up the project environment:**
pip install numpy pandas tensorflow scikit-learn imblearn seaborn matplotlib

**Usage**
Steps to utilize the project:
Data Preprocessing: Encode categorical variables and handle missing values.
Model Training: Train both SVM and Neural Network models using the preprocessed data.
Evaluation: Evaluate the models' performance through accuracy metrics and visualize the correlations and predictions.
Prediction: Deploy the models to predict depression and anxiety in new datasets.

Code Example
Example of model training:
python
# Training a Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
Contributing
Guidelines for contributing can be found in CONTRIBUTING.md.

# Authors
Kshitij Pathak - Initial work - Kshitijpathak22

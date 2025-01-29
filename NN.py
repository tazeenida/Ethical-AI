import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier


train_data = pd.read_csv("credit_train(in).csv")
test_data = pd.read_csv("credit_test(in).csv")

#preprocessing the data
X = train_data.drop(columns='approved')
y = train_data['approved']

#Normalizer
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)
test_data_normalized = normalizer.transform(test_data.drop(columns='approved'))

#split the data 
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

#initialize the model
mlp_model = MLPClassifier(random_state=42, max_iter=1100)

#model train
mlp_model.fit(X_train, y_train)

#make predictions
X_train_predictions = mlp_model.predict(X_train)
X_test_predictions = mlp_model.predict(X_test)

#evaluate the model
train_accuracy = accuracy_score(y_train, X_train_predictions)
test_accuracy = accuracy_score(y_test, X_test_predictions)

#print train and test accuracy
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay


train_data = pd.read_csv("credit_train(in).csv")
test_data = pd.read_csv("credit_test(in).csv")

#preprocessing the data
X = train_data.drop(columns='approved')
y = train_data['approved']

#normalize the data
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)
test_data_normalized = normalizer.transform(test_data.drop(columns='approved'))

#split the data 
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

#Initializing the logistic regression model
lr_model = LogisticRegression(random_state=42, max_iter=100)

#model training
lr_model.fit(X_train, y_train)

#make predictions
X_train_predictions = lr_model.predict(X_train)
X_test_predictions = lr_model.predict(X_test)

#Evaluate the model
train_accuracy = accuracy_score(y_train, X_train_predictions)
test_accuracy = accuracy_score(y_test, X_test_predictions)

#print train and test accuracy
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

#visualize the feature importance
coefficients = lr_model.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': train_data.drop(columns='approved').columns,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending= False)

#plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='skyblue')
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.gca().invert_yaxis()
plt.savefig("Feature_Importance_Logistic_Regression.png")

#Confusion Matrix Display
ConfusionMatrixDisplay.from_predictions(y_test, X_test_predictions, display_labels=['Not Approved', 'Approved'], cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("Logistic_Regression_Confusion_Matrix.png")

# generate report
train_report = classification_report(y_train, X_train_predictions)
test_report = classification_report(y_test, X_test_predictions)
train_conf_matrix = confusion_matrix(y_train, X_train_predictions)
test_conf_matrix = confusion_matrix(y_test, X_test_predictions)
with open("logistic_regression_report.txt", 'w') as f:
    f.write(f"Train Accuracy: {train_accuracy:.2f}\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.2f}\n\n")
    f.write("Train Classification Report:\n\n")
    f.write(train_report)
    
    f.write("\nTest Classification Report:\n\n")
    f.write(test_report)
    
    f.write("\nTrain Confusion Matrix: \n")
    f.write(str(train_conf_matrix))
    
    f.write("\n\nTest Confusion Matrix: \n")
    f.write(str(test_conf_matrix))


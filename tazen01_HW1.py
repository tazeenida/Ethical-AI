import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn import tree


#Load the dataset
train_data = pd.read_csv("credit_train(in).csv")
test_data = pd.read_csv("credit_test(in).csv")

#preprocessing the data
X = train_data.drop(columns='approved')
y = train_data['approved']

#split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#build the decision tree model
dt_model = DecisionTreeClassifier(random_state=42)

#Train the model
dt_model.fit(X_train, y_train)

#make predictions
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

#Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


#print train and test accuracy
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

#visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(dt_model, filled= True, feature_names=X.columns, class_names=['Not Approved', 'Approved'])
plt.savefig("decision_tree_plot.png")
tree.plot_tree(dt_model)


#visualize the feature importance
importances = dt_model.feature_importances_
feature_importance = list(zip(X.columns, importances))

for feature, importance in feature_importance:
    print(f"{feature}:{importance:.4f}")
features = [f[0] for f in feature_importance]
scores = [f[1] for f in feature_importance]

plt.figure(figsize=(10,6))
plt.barh(features, scores, color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Decision Tree")
plt.gca().invert_yaxis()
plt.savefig("Feature_Importance_Decision_Tree.png")

# generate report
train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
with open("decision_tree_report.txt", 'w') as f:
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

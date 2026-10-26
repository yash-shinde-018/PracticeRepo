
# Support Vector Machine using Iris Dataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

# i. Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# iii. Training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ii. Create and train SVM model
model = SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# iv. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("SVM - Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

# v. 5-fold cross-validation
pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear")
)

scores = cross_val_score(pipeline, X, y, cv=5)

print("\n5-Fold Cross-Validation Scores:")
print(scores)

print("Mean Cross-Validation Accuracy:", scores.mean())

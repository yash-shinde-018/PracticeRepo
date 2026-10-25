
# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

# Load the built-in Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# i. Data scaling and ii. Create the SVM model
# Pipeline scales data before fitting the SVM
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', random_state=42)
)

# iii. Training and testing the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# iv. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# v. Display 5-fold cross-validation score
cv_scores = cross_val_score(
    make_pipeline(StandardScaler(), SVC(kernel='linear')),
    X, y, cv=5, scoring='accuracy'
)

print("\n5-Fold Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

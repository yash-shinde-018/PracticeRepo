
# Import required libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the built-in Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# i. Data scaling
scaler = StandardScaler()

# ii. Create the Bagging classification model
model = make_pipeline(
    StandardScaler(),
    BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        random_state=42
    )
)

# iii. Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Display accuracy
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# iv. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=wine.target_names
))

# v. Display 5-fold cross-validation scores
cv_scores = cross_val_score(
    model, X, y, cv=5, scoring='accuracy'
)

print("\n5-Fold Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

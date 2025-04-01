import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
df = pd.read_csv("gym_data_noexp.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Separate features and target
X = df.drop("Workout_Type", axis=1)
y = df["Workout_Type"]

# Impute missing values with median (if needed)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split into train and test sets (70% train, 30% test, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print("Training class distribution:\n", y_train.value_counts(normalize=True))
print("\nTest class distribution:\n", y_test.value_counts(normalize=True))

# Standardize features (not strictly required for trees but useful for comparisons)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Random Forest Experiments
# -----------------------------

# Experiment 1: Random Forest model
rf_model1 = RandomForestClassifier(
    n_estimators=100, # Number of trees
    max_depth=5, # Maximum depth of each tree
    min_samples_leaf=2, # Minimum samples required to be at a leaf node
    max_features='sqrt',  # Number of features to consider when looking for the best split
    bootstrap=True, # Whether bootstrap samples are used when building trees
    random_state=42 # Random state for reproducibility
)
rf_model1.fit(X_train, y_train)
y_pred1 = rf_model1.predict(X_test)
print("Experiment 1 Accuracy:", accuracy_score(y_test, y_pred1))
print("Classification Report (Experiment 1):\n", classification_report(y_test, y_pred1))
print("Experiment 1 Train Accuracy:", rf_model1.score(X_train, y_train))
print("Experiment 1 Test Accuracy:", rf_model1.score(X_test, y_test))

# Experiment 2: More trees, more depth, and different min_samples_leaf
rf_model2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf_model2.fit(X_train, y_train)
y_pred2 = rf_model2.predict(X_test)
print("Experiment 2 Accuracy:", accuracy_score(y_test, y_pred2))
print("Classification Report (Experiment 2):\n", classification_report(y_test, y_pred2))
print("Experiment 2 Train Accuracy:", rf_model2.score(X_train, y_train))
print("Experiment 2 Test Accuracy:", rf_model2.score(X_test, y_test))

# Experiment 3:  Even more trees, no max depth, different min_samples_leaf
rf_model3 = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=3,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf_model3.fit(X_train, y_train)
y_pred3 = rf_model3.predict(X_test)
print("Experiment 3 Accuracy:", accuracy_score(y_test, y_pred3))
print("Classification Report (Experiment 3):\n", classification_report(y_test, y_pred3))
print("Experiment 3 Train Accuracy:", rf_model3.score(X_train, y_train))
print("Experiment 3 Test Accuracy:", rf_model3.score(X_test, y_test))

# Plot confusion matrix for Experiment 1
class_labels = ["Yoga", "HIIT", "Cardio", "Strength"]  # Adjust if necessary
cm = confusion_matrix(y_test, y_pred1)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Experiment 1)")
plt.show()


# Plot confusion matrix for Experiment 2
class_labels = ["Yoga", "HIIT", "Cardio", "Strength"]  # Adjust if necessary
cm = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Experiment 2)")
plt.show()


# Plot confusion matrix for Experiment 3
class_labels = ["Yoga", "HIIT", "Cardio", "Strength"]  # Adjust if necessary
cm = confusion_matrix(y_test, y_pred3)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Experiment 3)")
plt.show()

# Plot feature importances from Experiment 1
#importances = rf_model1.feature_importances_
#features = X.columns
#plt.figure(figsize=(10,6))
#sns.barplot(x=importances, y=features)
#plt.title("Feature Importances (Experiment 1)")
#plt.show()

# -----------------------------
# Hyperparameter Tuning with GridSearchCV
# -----------------------------
param_grid_rf = {
    "n_estimators": [100, 200, 300], # Number of trees
    "max_depth": [5, 10, None], 
    "min_samples_leaf": [1, 2, 3],
    "max_features": ['sqrt'], 
    "bootstrap": [True, False]
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring="accuracy"
)
grid_search_rf.fit(X_train, y_train)
print("Best Parameters (RF):", grid_search_rf.best_params_)
print("Best Cross-Validation Accuracy (RF):", grid_search_rf.best_score_)

best_rf_model = grid_search_rf.best_estimator_
y_pred_final_rf = best_rf_model.predict(X_test)
print("Final RF Model Test Accuracy:", accuracy_score(y_test, y_pred_final_rf))
print("Classification Report (Final RF Model):\n", classification_report(y_test, y_pred_final_rf))

# Optionally, save the best model
joblib.dump(best_rf_model, "rf_final_model.pkl")


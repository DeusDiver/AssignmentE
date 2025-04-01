import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("gym_data_noexp.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Separate features and target
# Replace "Workout_Type" with the exact name if different in your file.
X = df.drop("Workout_Type", axis=1)  # Features
y = df["Workout_Type"]               # Target


# Split the data: using 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Experiment 1:
log_reg1 = LogisticRegression(random_state=42, max_iter=1000)
log_reg1.fit(X_train_scaled, y_train)
y_pred1 = log_reg1.predict(X_test_scaled)

print("Experiment 1 Train Accuracy:", log_reg1.score(X_train_scaled, y_train))
print("Experiment 1 Accuracy:", accuracy_score(y_test, y_pred1))
print("Classification Report (Experiment 1):\n", classification_report(y_test, y_pred1))


# Experiment 2: Adjusting Regularization Strength (C)
log_reg2 = LogisticRegression(random_state=42, max_iter=1000, C=0.5)  # Stronger regularization
log_reg2.fit(X_train_scaled, y_train)
y_pred2 = log_reg2.predict(X_test_scaled)

print("Experiment 2 Train Accuracy:", log_reg2.score(X_train_scaled, y_train))
print("Experiment 2 Accuracy:", accuracy_score(y_test, y_pred2))
print("Classification Report (Experiment 2):\n", classification_report(y_test, y_pred2))

# Experiment 3: Different Regularization (L1 penalty)
# Note: For L1 penalty, solver must support it (e.g., 'liblinear' or 'saga')
log_reg3 = LogisticRegression(random_state=42, max_iter=1000, C=1.0, penalty='l1', solver='liblinear')
log_reg3.fit(X_train_scaled, y_train)
y_pred3 = log_reg3.predict(X_test_scaled)

print("Experiment 3 Train Accuracy:", log_reg3.score(X_train_scaled, y_train))
print("Experiment 3 Accuracy:", accuracy_score(y_test, y_pred3))
print("Classification Report (Experiment 3):\n", classification_report(y_test, y_pred3))

# --- Optional: Hyperparameter Tuning using GridSearchCV ---
param_grid = {
    'C': [0.1, 0.5, 1.0, 5.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # 'liblinear' works with both l1 and l2
}

grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000),
                           param_grid,
                           cv=5,
                           scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Final model evaluation using the best estimator
best_log_reg = grid_search.best_estimator_
y_pred_final = best_log_reg.predict(X_test_scaled)
print("Final Model Test Accuracy:", accuracy_score(y_test, y_pred_final))
print("Classification Report (Final Model):\n", classification_report(y_test, y_pred_final))

# Plot a confusion matrix for the final model
class_labels = ["Yoga", "HIIT", "Cardio", "Strength"]  # Adjust to your class names if needed
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

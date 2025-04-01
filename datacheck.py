import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("gym_data.csv")
df.columns = df.columns.str.strip()

# Display first few rows for reference
print("Data Snippet:")
print(df.head())

# -------------------------------------
# Descriptive Statistics and Summary
# -------------------------------------

# Calculate descriptive statistics for numerical features
desc_stats = df.describe().T  # Transpose for better readability
desc_stats['median'] = df.median()
desc_stats['mode'] = df.mode().iloc[0]
desc_stats['Q1'] = df.quantile(0.25)
desc_stats['Q3'] = df.quantile(0.75)

print("\nDescriptive Statistics:")
print(desc_stats)

# Print class distribution (target variable: Workout_Type)
print("\nClass Distribution:")
print(df["Workout_Type"].value_counts())

# -------------------------------------
# Visualization: Boxplots
# -------------------------------------

# List of features excluding the target
features = df.columns.drop("Workout_Type")

plt.figure(figsize=(15, 10))
for i, col in enumerate(features, 1):
    plt.subplot(3, 5, i)  # Adjust grid as needed based on number of features
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# -------------------------------------
# Visualization: Histograms
# -------------------------------------

plt.figure(figsize=(15, 10))
for i, col in enumerate(features, 1):
    plt.subplot(3, 5, i)
    sns.histplot(df[col], kde=True, bins=20)  # Histogram with KDE
    plt.title(col)
plt.tight_layout()
plt.show()

# -------------------------------------
# Visualization: Scatter Plots and Pairplot
# -------------------------------------
if "Session_Duration (hours)" in df.columns and "Calories_Burned" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, 
                    x="Session_Duration (hours)", 
                    y="Calories_Burned", 
                    hue="Workout_Type", 
                    palette="deep")
    plt.title("Session Duration vs. Calories Burned")
    plt.show()
else:
    print("Check column names for Session_Duration and Calories_Burned.")

# -------------------------------------
# Visualization: Correlation Matrix Heatmap
# -------------------------------------

corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

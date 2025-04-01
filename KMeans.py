import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data (using the same gym_data.csv)
df = pd.read_csv("gym_data_noexp.csv")
df.columns = df.columns.str.strip() 



# Drop the target variable for unsupervised learning
x = df.drop("Workout_Type", axis=1)

# Standardize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)



### K-Means Clustering Experiments ###

# Experiment with different k values and calculate silhouette scores
k_values = [2, 4, 6, 20, 100, 200] 
silhouette_scores = [] # Empty array

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)
    print(f"K-Means with k={k}: Silhouette Score = {score}")

# Plot silhouette scores vs. k values
plt.figure(figsize=(8,6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score for different k values")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

# For the chosen k (e.g., k=4 based on silhouette score), display cluster centers
kmeans_final = KMeans(n_clusters=4, init='k-means++', random_state=42)
clusters_final = kmeans_final.fit_predict(X_scaled)
print("Final K-Means Cluster Centers (scaled):\n", kmeans_final.cluster_centers_)
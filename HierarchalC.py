import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage


# Load and preprocess the data (using the same gym_data.csv)
df = pd.read_csv("gym_data_noexp.csv")
df.columns = df.columns.str.strip()

# Drop the target variable for unsupervised learning
x = df.drop("Workout_Type", axis=1)

# Standardize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

### Hierarchical Clustering Experiments ###

# Experiment 1: Agglomerative with 'ward' linkage
agglo1 = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agglo1 = agglo1.fit_predict(X_scaled)
print("Hierarchical Clustering (ward) Silhouette Score:", silhouette_score(X_scaled, labels_agglo1))

# Experiment 2: Agglomerative with 'complete' linkage
agglo2 = AgglomerativeClustering(n_clusters=4, linkage='complete')
labels_agglo2 = agglo2.fit_predict(X_scaled)
print("Hierarchical Clustering (complete) Silhouette Score:", silhouette_score(X_scaled, labels_agglo2))

# Experiment 3: Agglomerative with 'average' linkage
agglo3 = AgglomerativeClustering(n_clusters=4, linkage='average')
labels_agglo3 = agglo3.fit_predict(X_scaled)
print("Hierarchical Clustering (average) Silhouette Score:", silhouette_score(X_scaled, labels_agglo3))

# Optionally, visualize dendrogram (requires scipy linkage for full dendrogram)
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=12)
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()
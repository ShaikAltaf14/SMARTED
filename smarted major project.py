
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


data = pd.read_csv('customer_data.csv')


print(data.head())


data = data.dropna()


features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])


inertia = []  

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()


optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(8, 6))
plt.scatter(data['Age'], data['Annual Income (k$)'], c=data['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.colorbar(label='Cluster')
plt.show()

sil_score = silhouette_score(scaled_data, data['Cluster'])
print(f'Silhouette Score: {sil_score:.3f}')

cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
print("Cluster Centers:")
print(cluster_centers)


data.to_csv('customer_segmentation_with_clusters.csv', index=False)

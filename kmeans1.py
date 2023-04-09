import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


features=["HOSTEL_ID","HOSTEL_NAME","ADDRESS","CITY","COUNTRY","PINCODE","TYPE","RATING","LATITUDE","LONGITUDE","URL","PRICE","FACILITY","GENDER"]

        #reading csv file
df=pd.read_csv("Dataset.csv",names=features)

# Load and preprocess data
X = [[LATITUDE, LONGITUDE, RATING, PRICE] for LATITUDE, LONGITUDE, RATING, PRICE in df.iloc[["LATITUDE", "LONGITUDE", "RATING","PRICE"]]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters
# Use elbow method
elbow = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    elbow.append(kmeans.inertia_)
plt.plot(range(1, 11), elbow)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Or use silhouette score
from sklearn.metrics import silhouette_score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Score')
plt.show()

# Train KMeans model
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X_scaled)

# Make predictions on new data points
new_data = [[15.8876779, 75.7046777, 4.5, 4000]]
new_data_scaled = scaler.transform(new_data)
predicted_cluster = kmeans.predict(new_data_scaled)
predicted_value = kmeans.cluster_centers_[predicted_cluster]

# 0. Adjust parameters
NUM_CLUSTERS = 3      # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER =   5  # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris)

# 1. Import any other required libraries (e.g., numpy, scikit-learn)

from sklearn import datasets
import matplotlib.pyplot as plt


# 2. Load the Iris dataset using scikit-learn's load_iris() function
iris = datasets.load_iris()
data=iris.data


# 3. Implement K-Means Clustering
# 3.1. Import KMeans from scikit-learn
from sklearn.cluster import KMeans
# 3.2. Create an instance of KMeans with the specified number of clusters and max_iter
kmeans = KMeans(n_clusters=NUM_CLUSTERS, max_iter=MAX_ITER)
# 3.3. Fit the KMeans model to the data X
kmeans.fit(data)
# 3.4. Obtain the cluster labels
cluster_labels = kmeans.labels_


# 4. Visualize the Results
    # 4.1. Extract the features for visualization
x_feature = data[:, FEATURE_X_INDEX]  # Petal length
y_feature = data[:, FEATURE_Y_INDEX]  # Petal width

    # 4.2. Create a scatter plot of x_feature vs y_feature, colored by the cluster labels
plt.scatter(x_feature, y_feature, c=cluster_labels, cmap='viridis')
    # 4.3. Use different colors to represent different clusters
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, FEATURE_X_INDEX], centroids[:, FEATURE_Y_INDEX], s=300, c='red', marker='X', label='Centroids')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title(f'K-Means Clustering (n_clusters={NUM_CLUSTERS}, max_iter={MAX_ITER})')
plt.legend()
plt.show()
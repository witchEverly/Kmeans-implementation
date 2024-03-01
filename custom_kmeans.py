"""
Implementation K-Means clustering using object-oriented programming.

K-MEANS CLUSTERING:
1. Decide how many clusters you want, i.e. choose k
2. Randomly assign a centroid to each of the k clusters
3. Calculate the distance of all observation to each of the k centroids
4. Assign observations to the closest centroid
5. Find the new location of the centroids (mean of all observations in the cluster)
6. Repeat steps 3-5 until the centroids do not change position

https://domino.ai/blog/getting-started-with-k-means-clustering-in-python
"""
import numpy as np


class CustomKMeans:
    """
    KMeans clustering algorithm. The algorithm is initialized with the number of clusters and
    the maximum number of iterations. The algorithm is fit to the input data and the centroids
    are calculated. The algorithm can then be used to predict the cluster labels for new data.
    """

    def __init__(self, n_clusters=3, max_iterations=100, random_state=13):
        """
        Initialize KMeans clustering algorithm.

        Parameters:
        - n_clusters: Number of clusters (default is 3).
        - max_iterations: Maximum number of iterations (default is 100).
        - random_state: Seed for random number generation (default is None).
        """
        self.n_clusters = n_clusters  # Step 1 (set k clusters)
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, data):
        """
        Fit the KMeans algorithm to the input data.

        Parameters:
        - data: Numpy array of shape (m, n) representing m data points in an n-dimensional space.
        """
        # Step 2
        np.random.seed(self.random_state)
        self.centroids = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]
        # Step 3 and 4
        for i in range(self.max_iterations):
            labels = self.predict(data)
            # Step 5 and 6
            new_centroids = np.array(
                [data[labels == i].mean(axis=0) for i in range(self.n_clusters)]
            )
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, data, distance_metric='euclidean'):
        """
        Assign data points to the nearest cluster based on current centroids.

        Parameters:
        - data: Numpy array of shape (m, n) representing m data points in an n-dimensional space.

        Returns:
        - labels: Array of cluster labels assigned to each data point.
        """
        # Step 3
        if distance_metric == 'euclidean':
            distances = np.sqrt(((data - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        elif distance_metric == 'manhattan':
            distances = np.abs(data - self.centroids[:, np.newaxis]).sum(axis=2)
        else:
            raise ValueError('Invalid distance metric')

        # Step 4
        self.labels = np.argmin(distances, axis=0)

        return self.labels

    def get_centroids(self):
        """
        Get the current centroids after fitting the algorithm.

        Returns:
        - centroids: Numpy array representing the centroids of clusters.
        """
        return self.centroids

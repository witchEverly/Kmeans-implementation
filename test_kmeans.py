"""
Testing the KMeans class.
Blobs are generated using `make_blobs` for clustering.
"""
from custom_kmeans import CustomKMeans as KMeans
from sklearn.datasets import make_blobs
import numpy as np


def test_init():
    """
    Test 0: Check if the KMeans algorithm is initialized correctly.
    """
    kmeans = KMeans(n_clusters=3, max_iterations=100, random_state=13)
    assert kmeans.n_clusters == 3
    assert kmeans.max_iterations == 100
    assert kmeans.random_state == 13
    assert kmeans.centroids is None
    assert kmeans.labels is None


def test_kmeans_fit():
    """
    Test 1: Check if the centroids are initialized correctly,
    and the algorithm converges to the correct centroids.

    `make_blobs` generates data for clustering.
    """
    x, _ = make_blobs(n_samples=1000, centers=4, cluster_std=.7, random_state=0)
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(x)
    # print(kmeans.centroids)
    assert kmeans.centroids.shape == (4, 2)
    assert kmeans.labels.shape == (1000,)


def test_kmeans_predict():
    """
    Test 2: Check if the data points are assigned to the nearest cluster
    """
    x, _ = make_blobs(n_samples=1000, centers=4, cluster_std=.7, random_state=0)
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(x)
    labels = kmeans.predict(x)
    assert labels.shape == (1000,)
    assert len(np.unique(labels)) == 4
    labels = kmeans.predict(x, distance_metric='manhattan')
    assert labels.shape == (1000,)
    assert len(np.unique(labels)) == 4


def test_kmeans_get_centroids():
    """
    Test 3: Check if the centroids are updated correctly
    """
    x, _ = make_blobs(n_samples=1000, centers=4, cluster_std=.7, random_state=0)
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(x)
    centroids = kmeans.get_centroids()
    assert centroids.shape == (4, 2)

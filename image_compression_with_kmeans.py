"""
Image Compression using K-Means. The pixel space is mapped into $R^3$,
and the number of colors is reduced to compress the image.
"""
import matplotlib.pyplot as plt
from PIL import Image
from custom_kmeans import CustomKMeans as KMeans
import numpy as np


class ImageCompression:
    """Compress an image using K-Means clustering,
    and display the original and compressed images."""
    def __init__(self, n_clusters=8, max_iterations=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             max_iterations=self.max_iterations,
                             random_state=self.random_state)

    def compress(self, img_path):
        """
        Compress an image using K-Means clustering.
        :param img_path: Path to the image file.
        :return: img: Original image, compressed_img:
        Compressed image, num_clusters: Number of clusters
        """
        img = Image.open(img_path)
        img = np.array(img)
        img_data = img.reshape(-1, 3)
        self.kmeans.fit(img_data)
        cluster_centers = self.kmeans.get_centroids()
        labels = self.kmeans.predict(img_data)
        compressed_img = cluster_centers[labels].reshape(img.shape)

        return img, compressed_img, self.n_clusters

    @staticmethod   # Not connected to the instance of the class.
    def display(img, compressed_img, num_clusters):
        """
        Display the original and compressed images side by side.
        :param img: The original image
        :param compressed_img: The resulting compressed image from KMeans
        :param num_clusters: Number of clusters for plot title
        :return: None
        """
        ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[1].imshow(compressed_img.astype('uint8'))
        ax[1].set_title(f'Compressed Image, {num_clusters} colors')
        plt.show()

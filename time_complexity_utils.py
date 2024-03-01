"""
Time Complexity of KMeans, experiments to measure the running time for varying
values of parameters, and plots to visualize the growth in running time.
"""
import time
from custom_kmeans import CustomKMeans as KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class TimeComplexityKmeans:
    """Analyze the time complexity of the KMeans algorithm."""

    def __init__(self, n_clusters=3, max_iterations=100, random_state=13, kmeans=KMeans()):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.kmeans = kmeans

    def time_complexity(self,
                        values=None,
                        variable='m'):
        """
        Measure the running time for varying values of the parameters.
        :param values: List of values for the variable.
        :param variable: Variable to measure (m, n, k, or i):
        number of points (`m`), clusters (`K`), iterations (`I`), attributes (`n`).
        :return: times: List of running times for each value.
        """
        if values is None:
            values = [10, 100, 1000, 10000, 100000]

        times = []
        for value in values:
            if variable == 'm':
                x, _ = make_blobs(n_samples=value,
                                  centers=self.n_clusters,
                                  cluster_std=.7,
                                  random_state=self.random_state)
            elif variable == 'n':
                x, _ = make_blobs(n_samples=1000,
                                  centers=self.n_clusters,
                                  cluster_std=.7,
                                  n_features=value,
                                  random_state=self.random_state)
            elif variable == 'k':
                self.n_clusters = value
                x, _ = make_blobs(n_samples=1000,
                                  centers=self.n_clusters,
                                  cluster_std=.7,
                                  random_state=self.random_state)
            elif variable == 'i':
                self.max_iterations = value
                x, _ = make_blobs(n_samples=1000,
                                  centers=self.n_clusters,
                                  cluster_std=.7,
                                  random_state=self.random_state)
            else:
                raise ValueError('Invalid variable')

            start_time = time.time()
            self.kmeans.fit(x)
            times.append(time.time() - start_time)
        return times

    @staticmethod
    def plot_time_complexity(df,
                             variable,
                             title='KMeans Time Complexity'):
        """
        Create plots to visualize the growth in running time as each variable changes.
        :param df: DataFrame of running times.
        :param variable: Variable to plot (m, n, k, or i).
        :param title: Title for the plot.
        :return: None
        """
        plt.plot(df[variable], df['time'], marker='o')
        plt.title(f'{title}: {variable}')
        plt.xlabel(variable)
        plt.ylabel('Time (seconds)')

    @staticmethod
    def compare_distance_metrics(max_iterations=100, random_state=13):
        """Display the average running time for the KMeans algorithm,
        based on two distance metrics.
        :param max_iterations: Maximum number of iterations.
        :param random_state: Seed for random number generation.
        """
        X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=.7, random_state=random_state)
        time_elapsed_euc = 0
        time_elapsed_manhattan = 0
        for _ in range(max_iterations):
            k_means_euclidean = KMeans(n_clusters=5)
            start_time = time.time()
            k_means_euclidean.fit(X)
            end_time = time.time()
            time_elapsed_euc += end_time - start_time

            k_means_manhattan = KMeans(n_clusters=5)
            start_time = time.time()
            k_means_manhattan.fit(X)
            end_time = time.time()
            time_elapsed_manhattan += end_time - start_time

        print(f'Euclidean distance average: {time_elapsed_euc / max_iterations:.8f} seconds')
        print(f'Manhattan distance average: {time_elapsed_manhattan / max_iterations:.8f} seconds')

import numpy as np

class KMedoids:

    '''

    This class calculate K cluster with Kmeans method 

    ARGS:

    n_clusters: Number of groups to assign every observation of the data
    max_iters: Number maximun of iterations. This is a stop criteria
    X: Matrix with the data

    Return:

    Medoids: Center of each group of cluster
    Labels: Label of each data, according to the closest medoids
    '''

    def __init__(self, n_cluster, max_iterations=100):
        self.n_cluster = n_cluster
        self.max_iterations = max_iterations

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize medoids randomly from data points
        self.medoids_indices = np.random.choice(n_samples, self.n_cluster, replace=False)
        self.medoids = X[self.medoids_indices]

        for _ in range(self.max_iterations):
            # Assign each data point to the nearest medoid
            labels = self._assign_labels(X)

            # Calculate the total cost of the current clustering
            current_cost = self._calculate_cost(X, labels)

            # Try swapping each medoid with a non-medoid data point
            for i in range(self.n_cluster):
                non_medoids_indices = np.setdiff1d(np.arange(n_samples), self.medoids_indices)
                for j in non_medoids_indices:
                    new_medoids = self.medoids.copy()
                    new_medoids[i] = X[j]
                    new_labels = self._assign_labels(X, new_medoids)
                    new_cost = self._calculate_cost(X, new_labels)

                    # If the new configuration is better, update the medoids
                    if new_cost < current_cost:
                        self.medoids[i] = X[j]
                        labels = new_labels
                        current_cost = new_cost

        self.labels_ = labels
        return self

    def _assign_labels(self, X, medoids=None):
        if medoids is None:
            medoids = self.medoids
        # Assign each data point to the nearest medoid
        distances = np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _calculate_cost(self, X, labels):
        # Calculate the total cost of the current clustering
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
        cost = np.sum(distances[np.arange(len(X)), labels])
        return cost
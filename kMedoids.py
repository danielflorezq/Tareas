import numpy as np

class KMedoids:

    def __init__(self, n_clusters, max_iterations=100):
        self.n_cluster = n_clusters
        self.max_iterations = max_iterations


    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def initialize_medoids(self, X):
        #np.random.seed(1)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def assign_to_clusters(self, X, medoids):
        clusters = [[] for _ in range(self.n_clusters)]

        for point in X:
            distances = [self.euclidean_distance(point, medoid) for medoid in medoids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        return clusters

    def update_medoids(self, clusters):
        new_medoids = []

        for cluster in clusters:
            min_cost = float('inf')
            new_medoid = None

            for point in cluster:
                cost = sum(self.euclidean_distance(point, other_point) for other_point in cluster)
                if cost < min_cost:
                    min_cost = cost
                    new_medoid = point

            new_medoids.append(new_medoid)

        return new_medoids

    def fit(self, X):
        medoids = self.initialize_medoids(X)

        for _ in range(self.max_iterations):
            clusters = self.assign_to_clusters(X, medoids)
            new_medoids = self.update_medoids(clusters)

            if np.all(np.array_equal(medoids, new_medoids)):
                break

            medoids = new_medoids

        self.medoids_ = medoids
        self.labels_ = np.zeros(len(X), dtype=int)

        for i, cluster in enumerate(clusters):
            for point in cluster:
                self.labels_[np.where((X == point).all(axis=1))] = i

    def predict(self, X):
        distances = []

        for point in X:
            distances_to_medoids = [self.euclidean_distance(point, medoid) for medoid in self.medoids_]
            cluster_index = np.argmin(distances_to_medoids)
            distances.append(cluster_index)

        return np.array(distances)
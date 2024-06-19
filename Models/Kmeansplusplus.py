import numpy as np
import matplotlib.pyplot as plt 
# Función auxiliar para calcular la distancia Euclidiana entre dos puntos x1 y x2
def distance(x1, x2):
    return np.linalg.norm(x1 - x2, axis=0)
    
class KMeans_:
    def __init__(self, K=5, max_iters=1000, tol=1e-4, random_state=None, init='kmeans++'):
        self.K = K # Número de clusters
        self.max_iters = max_iters # Máximo número de iteraciones
        self.tol = tol  # Tolerancia para la convergencia
        self.random_state = random_state  # Semilla para la reproducibilidad
        self.init = init  # Método de inicialización ('random' o 'kmeans++')

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.X = X # Conjunto de datos
        self.n_samples, self.n_features = X.shape # Número de muestras y características

        # Inicializar los centroides
        if self.init == 'random':
            random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
            self.centroids = self.X[random_sample_idxs]
        elif self.init == 'kmeans++':
            self.centroids = self._initialize_centroids_kmeanspp()

        for _ in range(self.max_iters):
            # Asignar muestras a los centroides más cercanos (crear clusters)
            labels = self._assign_clusters(self.centroids)

            # Calcular nuevos centroides a partir de los clusters
            centroids_old = self.centroids
            self.centroids = self._calculate_centroids(labels)

            # Verificar convergencia
            if self._is_converged(centroids_old, self.centroids):
                break

    def predict(self, X):
        # Calcular distancias entre las muestras X y los centroides
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Devuelve el mas cercano
        return np.argmin(distances, axis=1)

    def _assign_clusters(self, centroids):
        # Calcular distancias entre todas las muestras y los centroides
        distances = np.linalg.norm(self.X[:, np.newaxis] - centroids, axis=2)
        # Devuelve el mas cercano
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, labels):
        centroids = np.zeros((self.K, self.n_features))
        for k in range(self.K):
            cluster_points = self.X[labels == k] # Muestras asignadas al cluster k
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)  # Calcular nuevo centroide como la media del cluster
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # Verificar si la suma de las distancias entre los centroides antiguos y nuevos es menor que la tolerancia
        distances = np.linalg.norm(centroids_old - centroids, axis=1)
        return np.all(distances < self.tol)
    
    def _initialize_centroids_kmeanspp(self):
        centroids = []
        # Paso 1: Seleccionar aleatoriamente el primer centroide de los puntos de datos
        first_centroid_idx = np.random.choice(self.n_samples)
        centroids.append(self.X[first_centroid_idx])
        for _ in range(1, self.K):
            # Paso 2: Calcular la distancia entre cada punto y el centroide más cercano
            distances = np.min([np.linalg.norm(self.X - c, axis=1) for c in centroids], axis=0)
            # Paso 3: Seleccionar el siguiente centroide con una probabilidad proporcional al cuadrado de la distancia
            probabilities = distances ** 2 / np.sum(distances ** 2)
            next_centroid_idx = np.random.choice(self.n_samples, p=probabilities)
            centroids.append(self.X[next_centroid_idx])
        
        return np.array(centroids)
    
    
def plot_scatter_kmeans(X, model, labels):
    #  Función para graficar un scatter plot de los datos junto con los centroides del modelo K-means.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    ax.scatter(model.centroids[:, 0], model.centroids[:, 1],
           marker='.', color='black', s=100 , linewidths=3)
    ax.set_title("K-means Clustering")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    plt.show()


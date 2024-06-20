import numpy as np
import matplotlib.pyplot as plt

class GaussianMixtureModel_:
    def __init__(self, k=3, epochs=1000, err=1e-5, random_state=None):
        self.k = k # Número de componentes gaussianas (clusters) en el modelo.
        self.epochs = epochs # Número máximo de iteraciones (épocas) para el entrenamiento.
        self.err = err # Error provocado a proposito para evitar overfitting
        self.random_state = random_state # Semilla para la reproducibilidad de los resultados aleatorios.
        self.centroids = None
        self.pi = None
        self.complete_covariance = None

    def fit(self, data):
        n_samples, n_features = data.shape

        # Initialize with KMeans++
 
        kmeans = KMeans_(K=self.k, random_state=self.random_state)

        kmeans.fit(data)
        self.centroids = kmeans.centroids

        self.pi = np.full(self.k, 1.0 / self.k)   # Inicializa los priors de forma uniforme
        self.complete_covariance = np.array([np.identity(n_features) for _ in range(self.k)])  # Inicializa las covarianzas
        max_log_likelihood = -np.inf
        best_centroids = self.centroids
        best_covariance = self.complete_covariance
        best_pi = self.pi

        for _ in range(self.epochs):
            # E-Step: Computar responsabilidades (gamma_nk)
            gaussian_result = np.zeros((self.k, n_samples))
            for i in range(self.k):
                gaussian_result[i, :] = self.multivariate_gaussian(data, self.centroids[i], self.complete_covariance[i])

            sum_nk = np.dot(self.pi, gaussian_result)
            gamma_nk = (self.pi[:, np.newaxis] * gaussian_result) / (sum_nk + self.err)

            # M-Step: Actualizar parametros
            self.pi = gamma_nk.sum(axis=1) / n_samples
            self.centroids = np.dot(gamma_nk, data) / gamma_nk.sum(axis=1)[:, np.newaxis]

            for i in range(self.k):
                diff = data - self.centroids[i]
                self.complete_covariance[i] = np.dot(gamma_nk[i] * diff.T, diff) / gamma_nk[i].sum()
                self.complete_covariance[i] += np.eye(n_features) * self.err

            # Log likelihood (A mas positivo mejor indica el progreso del modelo. Puede ser negativo)
            log_likelihood = np.sum(np.log(sum_nk + self.err))
            if log_likelihood > max_log_likelihood:
                max_log_likelihood = log_likelihood
                best_centroids = self.centroids
                best_covariance = self.complete_covariance
                best_pi = self.pi

        self.centroids = best_centroids
        self.complete_covariance = best_covariance
        self.pi = best_pi

    def multivariate_gaussian(self, data, mu, cov):
        n = data.shape[1]
        cov += np.eye(n) * self.err # Regularización para estabilidad numérica

        #Formula de stackoverflow + https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
        diff = data - mu
        inv_cov = np.linalg.inv(cov)
        norm_factor = np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))
        exp_factor = np.exp(-0.5 * np.sum(np.dot(diff, inv_cov) * diff, axis=1)) 
        return exp_factor / norm_factor

    def compute_gammank_gaussian(self, data):
        gaussian_result = np.zeros((self.k, len(data)))
        #Para cada gaussiana
        for i in range(self.k):
            #Calcular su resultado (meter el array dentro de una matriz 3D)
            gaussian_result[i, :] = self.multivariate_gaussian(data, self.centroids[i], self.complete_covariance[i])
        sum_nk = np.dot(self.pi, gaussian_result)
        gamma_nk = (self.pi[:, np.newaxis] * gaussian_result) / (sum_nk + self.err)
        return gamma_nk

    def predict(self, data):
        gamma = self.compute_gammank_gaussian(data)
        return np.argmax(gamma, axis=0)


def plot_gaussian_mixture(data, model, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)

    for i in range(model.k):
        plot_cov_ellipse(model.complete_covariance[i], model.centroids[i], nstd=2, alpha=0.5, edgecolor='red')

    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='white', marker='.', s=100, linewidths=3)
    plt.title('Distribución de Clases Predichas y Elipses de Covarianza')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.show()

def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
    ax = plt.gca()
    ax.add_artist(ellip)
class DBSCAN_:
    def __init__(self, radius=1, min_neighbors=5,random_state=None):
        self.radius = radius
        self.min_neighbors = min_neighbors
        self.labels_ = None  # Etiquetas finales asignadas a cada punto después del ajuste
        self.tree_ = None # Estructura de datos para búsqueda eficiente de vecinos
        self.data_ = None # Datos de entrada
        self.random_state = random_state  # Semilla para la reproducibilidad

    def fit(self, data):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.data_ = data
        self.tree_ = KDTree(data) # Construye un KDTree para búsqueda eficiente de vecinos
        n, _ = data.shape
        counter = 0
        labels = [-1 for _ in range(n)] # Inicializa todas las etiquetas como ruido (-1)
        label_count = -1 # Contador de etiquetas de clusters
        stack = [0] # Pila para seguir expandiendo clusters

        while True:
            if len(stack) == 0:
                try:
                    stack.append(labels.index(-1)) # Agrega puntos no etiquetados a la pila
                except ValueError:
                    self.labels_ = labels # Asigna las etiquetas finales al modelo
                    return self
            current_index = stack.pop() # Toma el siguiente punto de la pila

            if labels[current_index] == -1:
                counter += 1
                neighbors = self.tree_.query_radius([data[current_index]], r=self.radius)[0] #Busqueda por radio de vecinos cercanos
                n_neighbors = len(neighbors)
                if n_neighbors < self.min_neighbors:
                    labels[current_index] = -np.inf # Marca como ruido si no tiene suficientes vecinos
                else:
                    #Marcar index actual con el label
                    label_count += 1
                    labels[current_index] = label_count
                    for v in neighbors:
                        if labels[v] == -1:
                            stack.append(v)
                            labels[v] = label_count
                        if labels[v] == -np.inf:
                            labels[v] = label_count
            else:
                neighbors = self.tree_.query_radius([data[current_index]], r=self.radius)[0]
                for v in neighbors:
                    if labels[v] == -1:
                        counter += 1
                        stack.append(v)
                        labels[v] = label_count
                    if labels[v] == -np.inf:
                        labels[v] = label_count

        self.labels_ = labels
        return self

    def predict(self, data):
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet.")
        tree = KDTree(self.data_)
        labels = [-1 for _ in range(len(data))]
        for i, point in enumerate(data):
            neighbors = tree.query_radius([point], r=self.radius)[0]
            cluster_labels = [self.labels_[neighbor] for neighbor in neighbors if self.labels_[neighbor] >= 0]
            if cluster_labels:
                labels[i] = max(set(cluster_labels), key=cluster_labels.count)
        return np.array(labels)
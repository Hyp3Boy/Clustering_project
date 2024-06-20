# 🎥 Video-Based Human Action Recognition Project

## 🚀 Introduction
Welcome to our project on **Video-Based Human Action Recognition**! In this project, we analyze a dataset of videos depicting various human activities. Our goal is to extract relevant features, reduce dimensionality using advanced techniques, and apply clustering methods to group the videos by the actions performed.

## 👥 Authors
- **Josue Arbulú Pastor** (202210025)
- **Lama Carrasco Miguel** (201510170)
- **Santamaria Gutierrez Aaron** (202010399)
- **Chavez Zapata Lenin** (202210090)

## 📄 Abstract
We conducted an exploratory study on a dataset of human action videos to identify the most relevant features using non-linear dimensionality reduction techniques. Three clustering methods were applied to categorize the videos based on the actions performed.

## 🔑 Keywords
- **Clustering**
- **UMAP**
- **Gaussian Mixture Models**
- **K Means++**
- **R2+1**

## 📚 Dataset Overview
We used a subset of the **Human Action Recognition** dataset due to practical and computational constraints. Our dataset includes:
- **Training**: 5432 videos
- **Validation**: 427 videos
- **Testing**: 805 videos

### 🎬 Activities Included
- Dying hair
- Shot put
- Baking cookies
- Wrapping present
- Stretching leg
- Balloon blowing
- Riding camel
- Flipping pancake
- Fixing hair
- Spraying

## 🛠️ Feature Extraction
We utilized the `VideoFeatures` library to extract features using the R(2+1)D model due to its superior accuracy. Three configurations of R(2+1)D were employed to generate feature datasets for our analysis.

## 📉 Dimensionality Reduction
We compared two techniques for dimensionality reduction:
- **PCA** (Principal Component Analysis): A linear technique
- **UMAP** (Uniform Manifold Approximation and Projection): A non-linear technique

We also used **T-SNE** for better visualization of the clusters.

## 📊 Clustering Methods
We implemented and compared three clustering algorithms:
1. **K-means++**: Enhanced centroid initialization for better accuracy.
2. **Gaussian Mixture Model (GMM)**: Probabilistic clustering approach.
3. **DBSCAN**: Density-based clustering for handling noise and outliers.

## 🔍 Methodology
We conducted a grid search to optimize the combination of dataset, dimensionality reduction technique, model, and number of components. The optimization was based on silhouette score and adjusted rand index.

## 💻 Implementation
Our implementation is available in the [GitHub repository](https://github.com/Hyp3Boy/Clustering_project/tree/main). Check out the **models** folder for the code.

## 📈 Experimentation
We experimented with different configurations to find the best parameters for clustering. Below are some key results:

### Silhouette Scores
| Model Extraction | Dimension | Reduction | Model   | Silhouette |
|------------------|-----------|-----------|---------|------------|
| 34_32            | 5         | UMAP      | Kmeans++| 0.8424     |
| 34_32            | 5         | UMAP      | GMM     | 0.8388     |

### Rand Index
| Model Extraction | Dimension | Reduction | Model   | Rand Index |
|------------------|-----------|-----------|---------|------------|
| 34_32            | 3         | UMAP      | GMM     | 0.9448     |
| 34_32            | 6         | UMAP      | GMM     | 0.9446     |

## 📝 Conclusions
1. **Best Rand Index**: Gaussian Mixture Model performed best, indicating better alignment with the true structure of the data.
2. **Best Silhouette Score**: K-means++ excelled in maximizing cluster separation and cohesion.
3. **Data Preparation**: Using `PowerTransformer` and `UMAP` improved feature comparability and reduced data complexity.
4. **Model Choices**: Each model has its strengths:
    - **K-means++**: Simple, efficient for large datasets.
    - **GMM**: Handles complex data distributions well.
    - **DBSCAN**: Effective for noise and outliers.

## 📚 Future Work
- **Improve Class Distribution**: More balanced data for better clustering.
- **Evaluate Other Models**: Hierarchical clustering methods for detailed analysis.

## 📜 References
- [S3D: Single Shot multi-Span Detector](https://arxiv.org/abs/1711.11248)
- [Gaussian Mixture Models](https://arxiv.org/abs/2008.03679)
- [UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426)
- [k-means++: The advantages of careful seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)

Feel free to explore our project and contribute!

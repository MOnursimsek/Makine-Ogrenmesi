import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    """ Rastgele k tane merkez (centroid) seç """
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    return centroids

def assign_clusters(X, centroids):
    """ Her veri noktasını en yakın merkeze ata """
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(X, labels, k):
    """ Her merkezin konumunu güncelle """
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def kmeans(X, k, max_iters=100):
    """ K-means algoritmasını uygula """
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Örnek veri oluştur
np.random.seed(42)
X = np.vstack([np.random.multivariate_normal(mean, 0.1*np.eye(2), 50) for mean in [(0, 0), (1, 1), (2, 2)]])
k = 3

# Veriyi kümelenmeden önce çizdir
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', label='Veri Noktaları')
plt.title('Kümelenmeden Önce')
plt.legend()

# K-means algoritmasını çalıştır
centroids, labels = kmeans(X, k)

# Kümelenmiş veriyi çizdir
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Veri Noktaları')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Merkezler')
plt.title('Kümelendikten Sonra')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Çember şeklinde bir veri kümesi oluştur
np.random.seed(42)
theta = np.linspace(0, 2*np.pi, 100)
r = 10 + np.random.randn(100)
X_circle = np.array([r * np.cos(theta), r * np.sin(theta)]).T

# Veriyi çizdir (kümeleme yapılmadan önce)
plt.figure(figsize=(6, 6))
plt.scatter(X_circle[:, 0], X_circle[:, 1], c='gray')
plt.title('Kümelenmeden Önceki Veri')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')


# Bant genişliğini tahmin et
bandwidth = estimate_bandwidth(X_circle, quantile=0.2, n_samples=500)

# Mean Shift algoritmasını uygula
meanshift = MeanShift(bandwidth=bandwidth)
meanshift.fit(X_circle)

# Küme merkezlerini ve küme etiketlerini al
centroids = meanshift.cluster_centers_
labels = meanshift.labels_

# Sonuçları görselleştir
plt.figure(figsize=(6, 6))
plt.scatter(X_circle[:, 0], X_circle[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=100, label='Küme Merkezleri')
plt.title('Mean Shift Kümelenmesi')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.legend()
plt.show()

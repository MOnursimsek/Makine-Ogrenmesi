import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Örnek veri oluştur
np.random.seed(42)
X = np.vstack([np.random.multivariate_normal(mean, 0.1*np.eye(2), 50) for mean in [(-5, -5), (0, 0), (5, 5)]])
X = np.vstack([X, np.random.uniform(low=-7, high=7, size=(20, 2))])  # Gürültü ekle

# Veriyi kümeleme işlemi yapılmadan önce çizdir
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray')
plt.title('Kümelenmeden Önce')

# DBSCAN algoritmasını kullanarak kümeleme işlemi
eps = 0.5
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X)

# Kümeleme işlemi yapıldıktan sonra veriyi çizdir
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Kümelendikten Sonra')

plt.show()

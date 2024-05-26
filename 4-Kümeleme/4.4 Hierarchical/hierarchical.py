import numpy as np
import matplotlib.pyplot as plt

# Veri kümesi oluştur
np.random.seed(42)
X = np.random.rand(30, 2)

# Veriyi çizdir (hiyerarşik kümeleme yapılmadan önce)
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray')
for i in range(len(X)):
    plt.text(X[i, 0], X[i, 1], str(i), fontsize=12, ha='center', va='center')
plt.title('Hiyerarşik Kümeleme Öncesi Veri')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')


# Hiyerarşik kümeleme uygula
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, method='single')  # single linkage metodu kullanıldı

# Dendrogramı çizdir
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Veri Noktaları')
plt.ylabel('Uzaklık')
plt.show()

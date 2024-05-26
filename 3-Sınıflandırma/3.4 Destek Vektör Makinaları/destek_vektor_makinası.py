# Gerekli kütüphaneleri içe aktarın
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# California Housing veri kümesini yükleyin
housing = fetch_california_housing()

# Özellikleri ve hedef değişkeni ayırın
X = housing.data
y = (housing.target > 2.0).astype(int)  # Eşik değeri 2.0'dan büyükse 1, değilse 0

# Veri kümesini eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirin
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Destek vektör makineleri sınıflandırıcısı nesnesi oluşturun
svm_classifier = SVC(kernel='linear', random_state=42)

# Modeli eğitin
svm_classifier.fit(X_train_scaled, y_train)

# Test seti üzerinde tahmin yapın
y_pred = svm_classifier.predict(X_test_scaled)

# Doğruluk skorunu hesaplayın ve yazdırın
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

# Confusion matrisini oluşturun
conf_mat = confusion_matrix(y_test, y_pred)

# Confusion matrisini ısı haritası olarak çizdirin
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Confusion Matrix')
plt.show()

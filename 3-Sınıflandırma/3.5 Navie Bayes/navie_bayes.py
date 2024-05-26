# Gerekli kütüphaneleri içe aktarın
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Veri kümesini yükleyin (load_digits veri seti kullanılacak)
digits = load_digits()
X = digits.data
y = digits.target

# Veri kümesini eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes modelini oluşturun ve eğitin
model = GaussianNB()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred = model.predict(X_test)

# Doğruluk skorunu hesaplayın ve yazdırın
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

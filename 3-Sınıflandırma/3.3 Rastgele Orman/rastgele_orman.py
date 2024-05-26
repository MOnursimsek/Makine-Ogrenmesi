# Gerekli kütüphaneleri içe aktarın
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Pima Indian Diabetes veri setini yükleyin
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'class']
diabetes_data = pd.read_csv(url, names=column_names)

# Özellikler ve hedef değişkeni ayırın
X = diabetes_data.drop('class', axis=1)
y = diabetes_data['class']

# Veri kümesini eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rastgele Orman sınıflandırıcı nesnesi oluşturun
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Modeli eğitin
rf_classifier.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred = rf_classifier.predict(X_test)

# Doğruluk skorunu hesaplayın
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

# Confusion matrix oluşturun
conf_matrix = confusion_matrix(y_test, y_pred)

# Heatmap oluşturun
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Confusion Matrix')
plt.show()

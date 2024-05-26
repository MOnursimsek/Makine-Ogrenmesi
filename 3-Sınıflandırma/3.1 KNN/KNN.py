import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Veriyi yükleme
df = pd.read_csv("KNN/prostate.csv")
df.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 'Target' sütunu hariç diğer sütunları ölçeklendirme
scaler.fit(df.drop('Target', axis=1))
scaled_features = scaler.transform(df.drop('Target', axis=1))

# Ölçeklendirilmiş özellikleri yeni bir DataFrame'e koyma
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['Target'], test_size=0.30)

# Amacımız bir model oluşturup, bir kişinin 'Target' olup olmadığını tahmin etmek.
# k = 1 ile başlıyoruz.
for i in range(1,11,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    # Tahminler ve Değerlendirmeler
    # KNN modelimizi değerlendirelim
    print("k nın %d değeri için"%i)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

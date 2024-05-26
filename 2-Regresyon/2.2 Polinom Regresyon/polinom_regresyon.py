import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri setini içe aktarma
datas = pd.read_csv('Polinom Regresyon\data.csv')
datas

# Özellikler ve hedef değişkenlerin belirlenmesi
X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

# Lineer Regresyon modelini veri setine uyarlama
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X, y)

# Polinom Regresyon modelini veri setine uyarlama
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)  # Derecesi 4 olan polinom özellikleri oluşturma
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

# Lineer Regresyon sonuçlarını görselleştirme
plt.scatter(X, y, color='blue')  # Veri noktalarını mavi renkte çizme
plt.plot(X, lin.predict(X), color='red')  # Lineer Regresyon doğrusu (kırmızı)
plt.title('Lineer Regresyon')
plt.xlabel('Sıcaklık')
plt.ylabel('Basınç')
plt.show()

# Polinom Regresyon sonuçlarını görselleştirme
plt.scatter(X, y, color='blue')  # Veri noktalarını mavi renkte çizme
plt.plot(X, lin2.predict(poly.fit_transform(X)), color='red')  # Polinom Regresyon eğrisi (kırmızı)
plt.title('Polinom Regresyon')
plt.xlabel('Sıcaklık')
plt.ylabel('Basınç')
plt.show()

# Lineer Regresyon ile yeni bir sonuç tahmin etme (tahmin değişkenini 2D diziye dönüştürdükten sonra)
pred = 110.0
predarray = np.array([[pred]])
lin.predict(predarray)

# Polinom Regresyon ile yeni bir sonuç tahmin etme (tahmin değişkenini 2D diziye dönüştürdükten sonra)
pred2 = 110.0
pred2array = np.array([[pred2]])
lin2.predict(poly.fit_transform(pred2array))

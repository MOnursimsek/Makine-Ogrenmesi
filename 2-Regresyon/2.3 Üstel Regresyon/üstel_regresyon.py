import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Örnek veri oluşturma
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.5, 3.2, 4.0, 5.1, 6.5, 7.4, 9.2, 10.3, 12.7, 15.1])

# Üstel fonksiyon tanımı
def exponential_func(x, a, b):
    return a * np.exp(b * x)

# Üstel regresyonu uygula
popt, pcov = curve_fit(exponential_func, x, y, p0=(1, 0.1))

# Katsayıları çıkar
a, b = popt
print(f"Katsayılar: a = {a}, b = {b}")

# Modelin tahminlerini hesapla
y_pred = exponential_func(x, a, b)

# Gerçek veriler ve model tahminlerini görselleştir
plt.scatter(x, y, label='Gerçek Veriler', color='blue')
plt.plot(x, y_pred, label='Üstel Regresyon Modeli', color='red')
plt.xlabel('Bağımsız Değişken (x)')
plt.ylabel('Bağımlı Değişken (y)')
plt.legend()
plt.title('Üstel Regresyon')
plt.show()

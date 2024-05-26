# Gerekli paketleri içe aktarma
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Veri kümesini içe aktarma fonksiyonu
def veri_import_et():
    denge_verisi = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-' +
        'databases/balance-scale/balance-scale.data',
        sep=',', header=None)

    # Veri kümesi bilgilerini görüntüleme
    print("Veri Kümesi Uzunluğu: ", len(denge_verisi))
    print("Veri Kümesi Şekli: ", denge_verisi.shape)
    print("Veri Kümesi: ", denge_verisi.head())
    
    return denge_verisi

# Veri kümesini özellikler ve hedef değişkenler olarak ayırma fonksiyonu
def veri_setini_ayir(denge_verisi):

    # Hedef değişkeni ayırma
    X = denge_verisi.values[:, 1:5]
    Y = denge_verisi.values[:, 0]

    # Veri kümesini eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test

# Gini indeksi kullanarak model eğitimi
def gini_ile_egit(X_train, X_test, y_train):

    # Sınıflandırıcı nesnesi oluşturma
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    # Eğitimi gerçekleştirme
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Entropi kullanarak model eğitimi
def entropi_ile_egit(X_train, X_test, y_train):

    # Entropi ile karar ağacı
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    # Eğitimi gerçekleştirme
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Tahmin yapma fonksiyonu
def tahmin(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Tahmin Edilen Değerler:")
    print(y_pred)
    return y_pred

# Doğruluk hesaplama fonksiyonu
def dogrulugu_hesapla(y_test, y_pred):
    print("Karışıklık Matrisi: ",
          confusion_matrix(y_test, y_pred))
    print("Doğruluk : ",
          accuracy_score(y_test, y_pred)*100)
    print("Rapor : ",
          classification_report(y_test, y_pred))

# Karar ağacını görselleştirme fonksiyonu
def karar_agacini_gorsellestir(clf_object, ozellik_isimleri, sinif_isimleri):
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=ozellik_isimleri, class_names=sinif_isimleri, rounded=True)
    plt.show()

if __name__ == "__main__":
    veri = veri_import_et()
    X, Y, X_train, X_test, y_train, y_test = veri_setini_ayir(veri)

    clf_gini = gini_ile_egit(X_train, X_test, y_train)
    clf_entropy = entropi_ile_egit(X_train, X_test, y_train)

    # Karar Ağaçlarını Görselleştirme
    karar_agacini_gorsellestir(clf_gini, ['X1', 'X2', 'X3', 'X4'], ['L', 'B', 'R'])
    karar_agacini_gorsellestir(clf_entropy, ['X1', 'X2', 'X3', 'X4'], ['L', 'B', 'R'])

# Operasyon Aşaması
print("Gini İndeksi Kullanarak Sonuçlar:")
y_pred_gini = tahmin(X_test, clf_gini)
dogrulugu_hesapla(y_test, y_pred_gini)

print("Entropi Kullanarak Sonuçlar:")
y_pred_entropy = tahmin(X_test, clf_entropy)
dogrulugu_hesapla(y_test, y_pred_entropy)

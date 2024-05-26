import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

# CSV dosyasını oku
veri = pd.read_csv("Doğrusal regeresyon/data_for_lr.csv")
veri

# Eksik değerleri düşür
veri = veri.dropna()

# Eğitim veri seti ve etiketler
egitim_girdisi = np.array(veri.x[0:500]).reshape(500, 1)
egitim_cikti = np.array(veri.y[0:500]).reshape(500, 1)

# Doğrulama veri seti ve etiketler
test_girdisi = np.array(veri.x[500:700]).reshape(199, 1)
test_cikti = np.array(veri.y[500:700]).reshape(199, 1)


class DoğrusalRegresyon: 
    def __init__(self): 
        self.parametreler = {} 

    def ileri_yayılım(self, egitim_girdisi): 
        m = self.parametreler['m'] 
        c = self.parametreler['c'] 
        tahminler = np.multiply(m, egitim_girdisi) + c 
        return tahminler 

    def hata_fonksiyonu(self, tahminler, egitim_cikti): 
        hata = np.mean((egitim_cikti - tahminler) ** 2) 
        return hata 

    def geri_yayılım(self, egitim_girdisi, egitim_cikti, tahminler): 
        turevler = {} 
        df = (tahminler - egitim_cikti) 
        # dm= 2/n * (tahminler - gerçek) * girdi 
        dm = 2 * np.mean(np.multiply(egitim_girdisi, df)) 
        # dc = 2/n * (tahminler - gerçek) 
        dc = 2 * np.mean(df) 
        turevler['dm'] = dm 
        turevler['dc'] = dc 
        return turevler 

    def parametreleri_güncelle(self, turevler, öğrenme_oranı): 
        self.parametreler['m'] = self.parametreler['m'] - öğrenme_oranı * turevler['dm'] 
        self.parametreler['c'] = self.parametreler['c'] - öğrenme_oranı * turevler['dc'] 

    def eğit(self, egitim_girdisi, egitim_cikti, öğrenme_oranı, tekrar_sayısı): 
        # Rastgele parametrelerin başlatılması 
        self.parametreler['m'] = np.random.uniform(0, 1) * -1
        self.parametreler['c'] = np.random.uniform(0, 1) * -1

        # Kayıp fonksiyonunun başlatılması 
        self.kayip = [] 

        # Animasyon için şekil ve ekseni başlatma 
        fig, ax = plt.subplots() 
        x_değerleri = np.linspace(min(egitim_girdisi), max(egitim_girdisi), 100) 
        çizgi, = ax.plot(x_değerleri, self.parametreler['m'] * x_değerleri +
                        self.parametreler['c'], color='red', label='Regresyon Çizgisi') 
        ax.scatter(egitim_girdisi, egitim_cikti, marker='o', 
                color='green', label='Eğitim Verisi') 

        # Negatif değerleri dışlamak için y-ekseni sınırlarını ayarlama 
        ax.set_ylim(0, max(egitim_cikti) + 1) 

        def güncelle(frame): 
            # İleri yayılım 
            tahminler = self.ileri_yayılım(egitim_girdisi) 

            # Hata fonksiyonu 
            hata = self.hata_fonksiyonu(tahminler, egitim_cikti) 

            # Geri yayılım
            turevler = self.geri_yayılım(egitim_girdisi, egitim_cikti, tahminler) 

            # Parametreleri güncelle 
            self.parametreleri_güncelle(turevler, öğrenme_oranı) 

            # Regresyon çizgisini güncelle 
            çizgi.set_ydata(self.parametreler['m'] 
                        * x_değerleri + self.parametreler['c']) 

            # Kayıp değerini ekleyin ve yazdırın 
            self.kayip.append(hata) 
            print("Iterasyon = {}, Kayıp = {}".format(frame + 1, hata)) 

            return çizgi, 
        # Animasyon oluştur 
        ani = FuncAnimation(fig, güncelle, frames=tekrar_sayısı, interval=200, blit=True) 

        # Animasyonu bir video dosyası olarak kaydet (örneğin, MP4) 
        ani.save('doğrusal_regresyon_A.gif', writer='ffmpeg') 

        plt.xlabel('Giriş') 
        plt.ylabel('Çıkış') 
        plt.title('Doğrusal Regresyon') 
        plt.legend() 
        plt.show() 

        return self.parametreler, self.kayip 

# Örnek kullanım
doğrusal_regresyon = DoğrusalRegresyon()
parametreler, kayıp = doğrusal_regresyon.eğit(egitim_girdisi, egitim_cikti, 0.00001, 60)

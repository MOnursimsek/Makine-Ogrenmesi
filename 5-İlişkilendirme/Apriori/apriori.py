import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Örnek veri seti
dataset = [['ekmek', 'süt', 'peynir'],
           ['ekmek', 'süt', 'dondurma'],
           ['ekmek', 'süt', 'meyve'],
           ['ekmek', 'süt', 'yumurta'],
           ['ekmek', 'dondurma', 'meyve'],
           ['ekmek', 'meyve', 'peynir'],
           ['süt', 'meyve', 'peynir'],
           ['ekmek', 'süt', 'dondurma', 'meyve'],
           ['ekmek', 'soda']]

# Veri setini uygun formata dönüştürme
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Frekans tablosunu oluşturma
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Kuralları çıkar
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Sonuçları görüntüleme
print("Sık Görülen Ürün Kümeleri:\n", frequent_itemsets)
print("\nBirliktelik Kuralları:\n", rules)


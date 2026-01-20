import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
import datetime as dt
from pandas.core.interchange import dataframe



#-----------
#Settings
#-----------

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings("ignore")

#----------
# Data İmport
#----------

df = pd.read_csv("data/hotel_bookings.csv")

# children null olanları 0 olarak dolduruldu.
df["children"] = df["children"].fillna(0).astype(int)

# contry null olanlara "unknown" olarak dolduruldu.
df["country"] = df["country"].fillna("Unknown")

# agent null olanlara 0 olarak dolduruldu.
df["agent"] = df["agent"].fillna(0).astype(int)

# company null olanlara 0 olarak dolduruldu.
df["company"] = df["company"].fillna(0).astype(int)

# yetişkin bebek çocuk sayısının toplamının 0 olduğu rezervasyonları kaldırdık
danger_value = df[(df["adults"] + df["children"] + df["babies"]) == 0]
print(danger_value.shape[0]) #180 değer çıktı
df.drop(danger_value.index, inplace=True)

# reservation_status_date formatını tarih formatına değiştirdik
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])

# describe ına bakalım
df.describe([0.05, 0.25, 0.50, 0.75, 0.99,]).T

df = df[df["adr"] > 0]
df = df[df["adr"] < 5000]

df = df[df["adults"] < 10 ]
df = df[df["children"] < 5 ]
df = df[df["babies"] < 5 ]

# duplicete kontrolü
duplicate_sayisi = df.duplicated().sum() #31813 adet duplike satır var ve bunlardan kurtuluyoruz.

df.drop_duplicates(inplace=True) #85582 satırmız kaldı

df.columns = df.columns.str.strip()
# Feature Engineering

# 1. Toplam Kalış Süresi
df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

# 2.Odada toplam kalacak kişi sayısı
df['total_people'] = df['adults'] + df['children'] + df['babies']

# 3.Ciro(Toplam bir oda ve müşteriden kazanılan para)
df['revenue'] = df['adr'] * df['total_stay']

# 4. Aile Var mı Yok mu
df['is_family'] = 0
df.loc[(df['children'] > 0) | (df['babies'] >0), 'is_family'] = 1

# 5. Sadakat Kontrolü (daha önce iptal edilmiş rezervasyonu varsa bir de gerçekleşmiş konaklama varsa bunların toplamı)
df['total_customer_history'] = df['previous_cancellations'] + df['previous_cancellations']



#--------------------------------------------------
# Grab_col_names ile değişken kategorisi
#--------------------------------------------------
def grab_col_names(dataframe, cat_th=25, car_th=40):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

   ------

            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri
                    Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

         Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'O']

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                    dataframe[col].dtypes != "0"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "0"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car ]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat ]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols

cat_cols , cat_but_car , num_cols = grab_col_names(df)


#----------
# ANALİZ
#----------

# Genel Durum
print("Genel İptal Etme Durumu")
print(df["is_canceled"].value_counts(normalize=True) * 100)
print("-" * 30)





















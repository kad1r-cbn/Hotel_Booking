from statistics import quantiles

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
df['total_customer_history'] = df['previous_cancellations'] + df['previous_bookings_not_canceled']



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

# Kategorik Analiz
def target_summary_with_cat(dataframe, target, categorical_cols_):
    print(f"###########{categorical_cols_}###########")
    print(pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_cols_)[target].mean(),
        "Count": dataframe[categorical_cols_].value_counts(),
        "Ratio": dataframe[categorical_cols_].value_counts() / len(dataframe)
    }), end='\n\n\n')

#
for col in ["market_segment", "deposit_type", "is_family", "customer_type"]:
    target_summary_with_cat(df, "is_canceled", col)


# NUMERİC ANALİZ

analiz_listesi = num_cols + ['is_canceled']

analiz_listesi = [col for col in analiz_listesi if col in df.columns]

corr_matrix = df[analiz_listesi].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix,
            annot=True, #kutuların içine sayıları yaz
            fmt=".2f",  #virgülden sonra 2 basamak
            cmap="RdBu", #Kırmızı Mavi Skala
            vmin=-1 , vmax=1,) # -1 ile +1 arası değerlendirme.

plt.title("Korelasyon Haritası: Sayısal Değişkenler & İptal Durumu")
plt.show()

print(f"Sayısal Değişkenler: {num_cols}")
plt.figure(figsize=(20, 15))
for i, col in enumerate(num_cols):
    plt.subplot(4, 4, i+1) # Izgara boyutunu değişken sayısına göre ayarla (4x4 16 değişken alır)
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.tight_layout()

plt.show()
print(df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T)

# Yüzdelik Dilimlere  bak
# %99'da değer 200 iken Max 510 ise, 510 gerçekten uçtadır.
# Ama %99 zaten 450 ise, 510 normaldir.
print("\n--- Percentiles (Yüzdelik Dilimler) ---")
print(df[['lead_time', 'adr']].quantile([0.01, 0.05, 0.50, 0.95, 0.99]))

#  Lead Time 700+ olanlar kim? İptal mi etmişler?
print("\n--- Lead Time > 600 olanların durumu ---")
high_lead_time = df[df['lead_time'] > 600]
print(high_lead_time['is_canceled'].value_counts())


# Aykırı Değer Hesaplama ve Uygulama
def replace_with_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquartile = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquartile
    low_limit = quartile1 - 1.5 * interquartile

    # Baskılama
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
    if low_limit > 0:
        dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit

    print(f"{variable} için baskılama yapıldı. Üst Sınır: {up_limit:.2f}")

replace_with_thresholds(df, "lead_time") #max 750 --> 640
replace_with_thresholds(df, "adr") # max 510 --->450

# Ek Değişkenler

# İptal Oranı
df["cancellation_ratio"] = 0
mask = df["total_customer_history"] > 0
df.loc[mask,"canellation_ratio"] = df.loc[mask, "previous_cancellations"] / df.loc[mask, "total_customer_history"]

#lead time a göre müşteri segmentasyonu
def segment_lead_time(days):
    if days < 7:
        return "Last_Minute"
    elif days <= 30:
        return "Planner"
    else:
        return "Early_Bird"

df["lead_time_segment"] = df["lead_time"].apply(segment_lead_time)

# oda değişikliği talebi
df["room_change"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)

# --- YENİ ÖZELLİKLERİN DOĞRULAMASI (VALIDATION) ---

# 1. Segmentlere Göre İptal Oranları
# Bakalım Lead Time segmentasyonu işe yaramış mı?
print("--- Lead Time Segmentine Göre İptal Oranları ---")
print(df.groupby("lead_time_segment")["is_canceled"].mean().sort_values(ascending=False))

# 2. Oda Değişikliği Yapanların Durumu
# Teori: Oda değiştiren iptal etmez. Bakalım veri ne diyor?
print("\n--- Oda Değişikliği (Room Change) İptal Oranı ---")
print(df.groupby("room_change")["is_canceled"].mean())

# 3. Aileler mi bekarlar mı daha sadık?
print("\n--- Aile Durumuna Göre İptal Oranı ---")
print(df.groupby("is_family")["is_canceled"].mean())

# 4. İptal Oranı (Cancellation Ratio) Analizi
# Daha önce iptal etmiş olanların, şimdiki iptal durumu nedir?
# Bunu korelasyonla görelim
print("\n--- Sayısal Değişkenlerin Hedefle Korelasyonu ---")
new_features = ["total_customer_history", "cancellation_ratio", "total_stay", "total_people", "is_canceled"]
print(df[new_features].corr()["is_canceled"].sort_values(ascending=False))
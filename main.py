import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
import datetime as dt




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

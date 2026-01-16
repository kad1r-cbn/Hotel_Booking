from pyforest import *
import warnings
df= pd.read_csv('data/hotel_bookings.csv')
df_copy = df.copy()
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings("ignore")


df.head()
df.info()
df.shape
df.isnull().sum()
df.describe().T


def degisken_ozet(dataframe, cat_th=20, car_th=30):
    """
    Veri setindeki değişkenleri tiplerine göre ayırır ve listeler.
    cat_th: Sayısal ama kategorik olanlar için eşik değer (Örn: 10'dan az çeşidi olan sayılar)
    car_th: Kategorik ama çok fazla çeşidi olanlar (Kardinal) için eşik değer (Örn: İsimler, Tarihler)
    """

    # 1. KATEGORİK OLANLAR (Zaten category veya object olup, eşik değerden az olanlar)
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]
                and dataframe[col].nunique() < car_th]

    # 2. SAYISAL GÖRÜNÜMLÜ KATEGORİKLER (Sayısal olup, eşik değerden az olanlar - Örn: is_canceled)
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]
                   and dataframe[col].nunique() < cat_th]

    # Kategorikleri birleştir
    cat_cols = cat_cols + num_but_cat

    # 3. KARDİNAL OLANLAR (Object olup, eşik değerden fazla olanlar - Örn: Tarih, Ülke, İsim)
    cat_but_car = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object"]
                   and dataframe[col].nunique() > car_th]

    # 4. NUMERİK (Sayısal olup, kategorik olmayanlar - Örn: adr, lead_time)
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]
                and col not in cat_cols]

    print(f"Gözlem Sayısı: {dataframe.shape[0]}")
    print(f"Değişken Sayısı: {dataframe.shape[1]}")
    print(f"Kategorik Değişkenler (Toplam {len(cat_cols)}): {cat_cols}")
    print(f"Sayısal Değişkenler (Toplam {len(num_cols)}): {num_cols}")
    print(f"Kardinal Değişkenler (Yüksek Çeşitlilik - Toplam {len(cat_but_car)}): {cat_but_car}")

    return cat_cols, num_cols, cat_but_car


# Fonksiyonu çalıştır
cat_cols, num_cols, cat_but_car = degisken_ozet(df)

# 1. SAYISAL DEĞİŞKENLER (Matematiksel işlem yapılabilir)
# Bunların ortalamasını alabilirsin, grafiğini çizebilirsin.
num_cols = [
    'lead_time',
    'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies',
    'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'days_in_waiting_list',
    'adr',
    'required_car_parking_spaces', 'total_of_special_requests',
    'arrival_date_week_number', 'arrival_date_day_of_month', 'arrival_date_year'
]

# 2. KATEGORİK DEĞİŞKENLER (Gruplama yapılabilir)
# Bunları 'category' tipine çevireceğiz.
cat_cols = [
    'hotel', 'meal', 'country',
    'market_segment', 'distribution_channel',
    'is_repeated_guest',
    'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'customer_type',
    'reservation_status',
    'agent', 'company'  # Acente ve Şirket ID'leri de kategoriktir!
]

# 3. HEDEF DEĞİŞKEN (Tahmin etmeye çalıştığımız)
target_col = ['is_canceled']

# Dönüştürme işlemini yapalım (Hafızayı rahatlatalım)
for col in cat_cols:
    # Eğer sütun sayısal görünüyorsa (agent gibi) önce string'e, sonra kategoriye çevir
    # Bu, '9.0' ile '9' karmaşasını önler.
    df[col] = df[col].astype(str).astype('category')

print("✅ Dönüştürme tamamlandı! Listeler hazır.")
df.info()


df['arrival_date_full'] = (df['arrival_date_year'].astype(str) + " " +
                           df['arrival_date_month'] + " " +
                           df['arrival_date_day_of_month'].astype(str))
df['arrival_date_full'] = pd.to_datetime(df['arrival_date_full'])

df.duplicated().sum()
df["meal"].value_counts()
df["adr"].max()
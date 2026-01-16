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

cat_cols, num_cols, cat_but_car = degisken_ozet(df)

# 1. SAYISAL DEĞİŞKENLER (Matematiksel işlem yapılabilir)
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
df.info()


df['arrival_date_full'] = (df['arrival_date_year'].astype(str) + " " +
                           df['arrival_date_month'] + " " +
                           df['arrival_date_day_of_month'].astype(str))
df['arrival_date_full'] = pd.to_datetime(df['arrival_date_full'])

df.duplicated().sum()
df["meal"].value_counts()
df["adr"].max()



# children null olanları 0 olarak dolduruldu.
df["children"] = df["children"].fillna(0).astype(int)

# contry null olanlara "unknown" olarak dolduruldu.
# 1. Adım: Önce kategori listesine 'Unknown' seçeneğini ekle
if 'Unknown' not in df['country'].cat.categories:
    df['country'] = df['country'].cat.add_categories('Unknown')

# 2. Adım: Şimdi gönül rahatlığıyla boşlukları doldurabilirsin
df['country'] = df['country'].fillna('Unknown')

# agent null olanlara 0 olarak dolduruldu.
# 1. Önce sütunları string (yazı) yapalım ki işlem garanti olsun
df["agent"] = df["agent"].astype(str)
df["company"] = df["company"].astype(str)

# 2. "nan" yazan yerleri "0" ile değiştirelim
df["agent"] = df["agent"].replace("nan", "0")
df["company"] = df["company"].replace("nan", "0")

# 3. Önce Float yapalım (Çünkü "9.0" yazısını direkt int yapamazsın, önce 9.0 ondalıklı sayı olmalı)
df["agent"] = df["agent"].astype(float).astype(int)
df["company"] = df["company"].astype(float).astype(int)

# 4. Son olarak Kategori yapıp paketleyelim
df["agent"] = df["agent"].astype("category")
df["company"] = df["company"].astype("category")# company null olanlara 0 olarak dolduruldu.
df["company"] = df["company"].fillna(0).astype(int)

# yetişkin bebek çocuk sayısının toplamının 0 olduğu rezervasyonları kaldırdık
danger_value = df[(df["adults"] + df["children"] + df["babies"]) == 0]
print(danger_value.shape[0]) #180 değer çıktı
df.drop(danger_value.index, inplace=True)

# reservation_status_date formatını tarih formatına değiştirdik
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])

# 1. Duplicate'leri Sil
df.drop_duplicates(inplace=True)

# 2. Negatif ve Aşırı Yüksek Fiyatları Temizle (0 ile 5000 arası kalsın)
# Not: Bedava (0) konaklamalar kalabilir, onlar promosyon olabilir.
df = df[(df["adr"] >= 0) & (df["adr"] < 5000)]

# 3. Hayalet ve "Otobüs" Misafirleri Temizle
# Hiç kimsenin kalmadığı (0 kişi) veya aşırı kalabalık (örn: 10 kişiden fazla) odaları atalım.
df = df[(df["adults"] + df["children"] + df["babies"] > 0)]
df = df[(df["adults"] + df["children"] + df["babies"] <= 10)]

# 4. Undefined Yemekleri Düzelt
df.loc[df["meal"] == "Undefined", "meal"] = "SC"

# --- RAPORLAMA ---
print("✅ Temizlik Tamamlandı.")
print(f"Kalan Satır Sayısı: {df.shape[0]}")

print(df['is_canceled'].value_counts(normalize=True))
df.head()
df.info()
df.shape
df.isnull().sum()
df.describe().T
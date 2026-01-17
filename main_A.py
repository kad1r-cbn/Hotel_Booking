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




outlier_candidates = [
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'lead_time',                    # Ne kadar erken rezervasyon yapıyorlar?
    'adr',                          # Fiyatlarda aşırı uçlar kaldı mı?
    'days_in_waiting_list',         # Bekleme listesinde çürüyenler var mı?
    'previous_cancellations',       # Seri iptalciler (Risk!)
    'total_of_special_requests',    # Çok aşırı istek yapanlar kim?
    'stays_in_week_nights'          # Otelde aylarca kalan var mı?
]
for col in outlier_candidates:
    plt.figure(figsize=(10, 2)) # Geniş ve kısa grafikler
    sns.boxplot(x=df[col], color="orange")
    plt.title(f"Aykırı Değer Analizi: {col}", fontweight="bold")
    plt.show()


df=df[df["babies"]<5]
df.shape[0]




channel_analysis = df.groupby('market_segment', observed=True).agg({
    'is_canceled': ['count', 'mean'],  # Hacim ve İptal Riski
    'adr': 'mean'                     # Kârlılık (Ortalama Fiyat)
})
channel_analysis.columns = ["Toplam Rezervasyon", "İptal Oranı", "Ortalama Fiyat (ADR)",]
channel_analysis["Pazar Payı (%)"] = (channel_analysis["Toplam Rezervasyon"] / channel_analysis["Toplam Rezervasyon"].sum()) * 100
print(channel_analysis.sort_values(by="Toplam Rezervasyon", ascending=False))




monthly_stats = df.groupby('arrival_date_month', observed=True).agg({
    'is_canceled': ['count', 'mean'],
    'adr': ['mean']
})

monthly_stats.columns = ["Rezervasyon Sayısı", "İptal Oranı", "Ortalama Fiyat"]
monthly_stats = monthly_stats.reindex(month_order)

print(monthly_stats)

plt.figure(figsize=(14, 6))
sns.barplot(x=monthly_stats.index, y=monthly_stats["Rezervasyon Sayısı"], color="skyblue", label="Rezervasyon Sayısı")
ax2 = plt.twinx()
sns.lineplot(x=monthly_stats.index, y=monthly_stats["İptal Oranı"], color="red", marker="o", lw=3, label="İptal Oranı", ax=ax2)
plt.title("Aylara Göre Doluluk ve İptal Riski Analizi", fontsize=16)
plt.show()


print(df[df["arrival_date_month"].isin(["April", "June", "December"])].groupby([
    "arrival_date_month",
    "market_segment"])["is_canceled"].agg([
    "count",
    "mean"]).sort_values(by=[
    "arrival_date_month", "mean"], ascending=[True, False]))




top_10_countries = df['country'].value_counts().head(10).index
country_analysis = df[df['country'].isin(top_10_countries)].groupby('country', observed=True).agg({
    'is_canceled': ['count', 'mean'],  # Sayı ve İptal Oranı
    'adr': 'mean'                      # Bıraktıkları Para
})
country_analysis.columns = ["Toplam Rezervasyon", "İptal Oranı", "Ortalama Fiyat (ADR)"]
country_analysis["Pazar Payı (%)"] = (country_analysis["Toplam Rezervasyon"] / len(df)) * 100
print(country_analysis.sort_values(by="Toplam Rezervasyon", ascending=False).round(2))
plot_data = country_analysis.sort_values(by="İptal Oranı", ascending=False).head(10)
plot_data.index = plot_data.index.astype(str)
plt.figure(figsize=(12, 6))
sns.barplot(x=plot_data.index, y=plot_data["İptal Oranı"], palette="viridis")
plt.title("En Yüksek İptal Oranına Sahip 10 Ülke", fontsize=14)
plt.ylabel("İptal Oranı")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 1. Misafir Tipi Sınıflandırması (Feature Engineering)
# Mantık: Eğer çocuk veya bebek varsa "Aile", yoksa ve 2 kişiyse "Çift", 1 kişiyse "Tek"
def classify_guest(row):
    if row['children'] > 0 or row['babies'] > 0:
        return 'Family (Aile)'
    elif row['adults'] == 2:
        return 'Couple (Çift)'
    elif row['adults'] == 1:
        return 'Single (Tek)'
    else:
        return 'Group/Other' # 3+ yetişkin vs.

# Yeni sütunu oluşturalım
df['guest_type'] = df.apply(classify_guest, axis=1)

# 2. Misafir Tiplerine Göre Analiz
guest_analysis = df.groupby('guest_type', observed=True).agg({
    'is_canceled': ['count', 'mean'],
    'adr': 'mean'
})

guest_analysis.columns = ["Toplam Rezervasyon", "İptal Oranı", "Ortalama Fiyat (ADR)"]
guest_analysis["Pazar Payı (%)"] = (guest_analysis["Toplam Rezervasyon"] / len(df)) * 100

print(guest_analysis.sort_values(by="Toplam Rezervasyon", ascending=False).round(2))


# 1. Önce sadece GERÇEKLEŞEN (İptal olmayan) konaklamaları alalım
real_guests = df[df['is_canceled'] == 0].copy()

# 2. Toplam Geceleme Sayısını Hesaplayalım (Hafta içi + Hafta sonu)
real_guests['total_nights'] = real_guests['stays_in_weekend_nights'] + real_guests['stays_in_week_nights']

# 3. Her Müşteriden Kazanılan TOPLAM PARAYI (Revenue) Hesaplayalım
# Formül: Gece Sayısı * Günlük Fiyat (ADR)
real_guests['total_revenue'] = real_guests['total_nights'] * real_guests['adr']

# --- ANALİZ 1: ÜLKELERE GÖRE KAZANÇ ---
country_revenue = real_guests.groupby('country', observed=True).agg({
    'total_revenue': 'sum',      # Kasaya giren toplam para
    'adr': 'mean',               # Ortalama oda fiyatı
    'total_nights': 'mean',      # Ortalama kaç gece kalıyorlar?
    'is_canceled': 'count'       # Kaç kişi gelmiş?
})

country_revenue.columns = ["Toplam Ciro (Revenue)", "Ortalama Fiyat (ADR)", "Ortalama Geceleme", "Misafir Sayısı"]
# Ciroya göre sırala ve ilk 10'u göster
print("-" * 30)
print("💰 ÜLKELERE GÖRE KAZANÇ LİDERLERİ")
print("-" * 30)
print(country_revenue.sort_values(by="Toplam Ciro (Revenue)", ascending=False).head(10).round(2))


# --- ANALİZ 2: PAZAR SEGMENTİNE GÖRE KAZANÇ ---
segment_revenue = real_guests.groupby('market_segment', observed=True).agg({
    'total_revenue': 'sum',
    'adr': 'mean'
})
segment_revenue.columns = ["Toplam Ciro (Revenue)", "Ortalama Fiyat (ADR)"]
print("\n" + "-" * 30)
print("🏨 KANALLARA GÖRE KAZANÇ LİDERLERİ")
print("-" * 30)
print(segment_revenue.sort_values(by="Toplam Ciro (Revenue)", ascending=False).round(2))



# --- CRM TEŞHİS ANALİZİ: LEAD TIME & DEPOSIT ---

# 1. LEAD TIME KATEGORİZASYONU (Müşteri Davranışını Anlamak İçin)
# Müşterileri "Planlılar" ve "Spontane Olanlar" diye ayıralım
bins = [0, 7, 30, 90, 180, 365, 730]
labels = ['Son Dakikacılar (0-7 Gün)', 'Yakın Plan (8-30 Gün)', 'Orta Vade (1-3 Ay)', 'Uzun Vade (3-6 Ay)', 'Çok Uzun (6-12 Ay)', 'Yıllık Plan (1+ Yıl)']

df['lead_time_segment'] = pd.cut(df['lead_time'], bins=bins, labels=labels)

# Lead Time Segmentlerine göre İptal Oranları
lead_time_analysis = df.groupby('lead_time_segment', observed=True)['is_canceled'].mean() * 100

# 2. DEPOSIT TİPİ ANALİZİ (Finansal Bağlılık)
# Parayı ödeyen gerçekten sadık kalıyor mu?
deposit_analysis = df.groupby('deposit_type', observed=True)['is_canceled'].mean() * 100

# 3. ÖZEL İSTEK ETKİSİ (Hizmet Beklentisi)
# Özel istekte bulunan müşteri, otelle bağ kurmuş demektir. İptal oranı düşük mü?
df['has_request'] = df['total_of_special_requests'] > 0
request_analysis = df.groupby('has_request', observed=True)['is_canceled'].mean() * 100

# --- SONUÇLARI YAZDIRALIM ---
print(f"{'-'*30}\n📊 BEKLEME SÜRESİNE (LEAD TIME) GÖRE İPTAL ORANLARI (%)\n{'-'*30}")
print(lead_time_analysis.round(2))

print(f"\n{'-'*30}\n💰 DEPOZİTO TİPİNE GÖRE İPTAL ORANLARI (%)\n{'-'*30}")
print(deposit_analysis.round(2))

print(f"\n{'-'*30}\n🛎️ ÖZEL İSTEK (SPECIAL REQUEST) ETKİSİ (%)\n{'-'*30}")
print(f"Özel İsteği Olmayanların İptal Oranı: %{request_analysis[False]:.2f}")
print(f"Özel İsteği Olanların İptal Oranı:    %{request_analysis[True]:.2f}")





# 1. VERİ HAZIRLIĞI
# Sadece gerçekleşen (iptal olmayan) rezervasyonları alıyoruz, çünkü iptal edenden para kazanmadık.
rfm_df = df[df['is_canceled'] == 0].copy()

# Analiz Tarihi (Verideki son tarihten 2 gün sonrası)
analysis_date = rfm_df['arrival_date_full'].max() + dt.timedelta(days=2)

# --- 2. METRİKLERİN HESAPLANMASI (HAM DEĞERLER) ---

# R (RECENCY): Müşteri kaç gün önce geldi?
rfm_df['Recency'] = (analysis_date - rfm_df['arrival_date_full']).dt.days

# F (FREQUENCY): Müşteri toplam kaç kez geldi?
# İPUCU: Veri setindeki 'previous_bookings_not_canceled' sütunu müşterinin geçmişini söyler.
# Buna +1 ekliyoruz (çünkü şu anki konaklaması da var).
rfm_df['Frequency'] = rfm_df['previous_bookings_not_canceled'] + 1

# M (MONETARY): Müşteri toplam ne kadar ödedi?
rfm_df['Monetary'] = rfm_df['adr'] * (rfm_df['stays_in_weekend_nights'] + rfm_df['stays_in_week_nights'])

# Negatif veya sıfır bedelli (Complementary) odaları temizleyelim ki skor bozulmasın
rfm_df = rfm_df[rfm_df['Monetary'] > 0]

# --- 3. SKORLAMA (1-5 ARASI PUAN VERME) ---

# Recency Score (5 = En Yeni, 1 = En Eski)
rfm_df["Recency_Score"] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])

# Frequency Score (5 = Çok Sık Gelen, 1 = Tek Seferlik)
# Not: Çoğu kişi 1 kere geldiği için burada yoğunluk 1'de toplanabilir, rank metoduyla zorluyoruz.
rfm_df["Frequency_Score"] = pd.qcut(rfm_df['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# Monetary Score (5 = Çok Para, 1 = Az Para)
rfm_df["Monetary_Score"] = pd.qcut(rfm_df['Monetary'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# --- 4. RFM SKORUNU BİRLEŞTİRME ---
# İşte senin aradığın "555", "121" gibi karne notları burada oluşuyor.
rfm_df["RFM_SCORE"] = (rfm_df['Recency_Score'].astype(str) +
                       rfm_df['Frequency_Score'].astype(str) +
                       rfm_df['Monetary_Score'].astype(str))

# --- 5. SEGMENTASYON (Müşteri Etiketleri) ---
# Segmentleri R ve F skorlarına göre belirleriz (Klasik RFM Yaklaşımı)
seg_map = {
    r'[1-2][1-2]': 'Uyuyanlar (Hibernating)',
    r'[1-2][3-4]': 'Riskli (At Risk)',
    r'[1-2]5': 'Kaybedilemez (Cant Loose)',
    r'3[1-2]': 'Uykuya Dalıyor (About to Sleep)',
    r'33': 'Dikkat (Need Attention)',
    r'[3-4][4-5]': 'Sadık Müşteriler (Loyal)',
    r'41': 'Umut Vaat Eden (Promising)',
    r'51': 'Yeni Gelen (New Customers)',
    r'[4-5][2-3]': 'Potansiyel Sadık (Potential Loyal)',
    r'5[4-5]': 'ŞAMPİYONLAR (Champions)'
}

# Regex ile skorları isme çevir (Sadece R ve F'ye bakarak)
rfm_df['Segment'] = (rfm_df['Recency_Score'].astype(str) + rfm_df['Frequency_Score'].astype(str)).replace(seg_map, regex=True)

# --- ÇIKTI 1: SENİN GÖRMEK İSTEDİĞİN DETAYLI TABLO ---
print(f"{'-'*60}\n📋 RFM ANALİZ TABLOSU (R, F, M Değerleri ve Skorları)\n{'-'*60}")
# Sütunları senin için seçiyorum: Ham değerler VE Skorlar yan yana
cols_to_show = ['country', 'market_segment',
                'Recency', 'Recency_Score',
                'Frequency', 'Frequency_Score',
                'Monetary', 'Monetary_Score',
                'RFM_SCORE', 'Segment']

print(rfm_df[cols_to_show].head(15))

# 1. Renk Matrisini Hazırlama (Kalite Skoru: R + F)
# Renkler artık kişi sayısına göre değil, skorun iyiliğine göre (Yeşil=5+5, Kırmızı=1+1) sabitlenecek.
r_labels = [5, 4, 3, 2, 1]
f_labels = [5, 4, 3, 2, 1]
quality_matrix = pd.DataFrame(
    [[r + f for r in r_labels] for f in f_labels],
    index=f_labels, columns=r_labels
)

# 2. Gerçek Veriyi (Sayıları) Hazırlama
rfm_count = rfm_df.groupby(['Frequency_Score', 'Recency_Score'], observed=True).size().unstack().reindex(index=f_labels, columns=r_labels).fillna(0)
rfm_labels = rfm_df.groupby(['Frequency_Score', 'Recency_Score'], observed=True)['Segment'].agg(lambda x: x.mode()[0]).unstack().reindex(index=f_labels, columns=r_labels).fillna("")

# 3. Etiketleri Oluşturma
clean_labels = rfm_labels.apply(lambda col: col.str.split('(').str[0]) # İngilizceyi temizle
annot_labels = clean_labels.astype(str) + "\n(" + rfm_count.astype(int).astype(str) + " Kişi)"

# 4. Çizim
plt.figure(figsize=(15, 9))
sns.heatmap(
    quality_matrix,     # RENKLER: Sabit Kalite Skoruna Göre (Yeşil=İyi, Kırmızı=Kötü)
    annot=annot_labels, # YAZILAR: Gerçek Kişi Sayıları
    fmt='',
    cmap='RdYlGn',      # Artık doğru çalışır (Skor yüksekse Yeşil)
    linewidths=2,
    linecolor='white',
    cbar=False,
    annot_kws={"size": 11, "weight": "bold", "color": "black"} # Siyah yazı her renkte okunur
)

plt.title("RFM Segment Analizi (Doğru Renklendirme)", fontsize=16)
plt.xlabel("Recency (Yenilik) Skoru", fontsize=12)
plt.ylabel("Frequency (Sıklık) Skoru", fontsize=12)
plt.show()

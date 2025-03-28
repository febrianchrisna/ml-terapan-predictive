# -*- coding: utf-8 -*-
"""mobil_bekas.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SxIShCz83Wki1VUaMss0uW786G8jAcSr

# Data Loading
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

# load the dataset
url = '/content/formattedData.csv'
cars = pd.read_csv(url)
cars

"""# EDA"""

cars.info()

"""Fitur Numerik
- Kilometres (Jarak tempuh kendaraan)
- Passengers (Jumlah penumpang yang bisa ditampung kendaraan)
- Doors (Jumlah pintu kendaraan)
- City (Lokasi kota kendaraan tersebut berada)
- Highway (Kondisi kendaraan di jalan tol)
- Price (Harga kendaraan)

Fitur Kategorikal:
- Year (Tahun kendaraan)
- Make (Merek kendaraan)
- Model (Model kendaraan)
- Engine (Jenis mesin kendaraan)
- Body_Type (Tipe bodi kendaraan)
- Transmission (Jenis transmisi kendaraan)
- Drivetrain (Jenis penggerak kendaraan)
- Exterior_Colour (Warna eksterior kendaraan)
- Interior_Colour (Warna interior kendaraan)
- Fuel_Type (Jenis bahan bakar kendaraan)

## Kesalahan Tipe Data
"""

cars['Engine'].unique()

"""Pada kolom Engine, terdapat kesalahan tipe data dimana nilai string 'E' muncul, padahal seharusnya nilai tersebut adalah 0, yang menunjukkan bahwa mobil tersebut tidak memiliki mesin berbahan bakar fosil atau merupakan mobil listrik. Untuk memperbaiki hal ini, nilai 'E' akan diganti dengan 0, dan tipe data kolom tersebut telah diubah menjadi integer agar lebih sesuai dengan format yang diinginkan dan dapat digunakan dalam analisis lebih lanjut."""

cars['Engine'] = cars['Engine'].replace('E', 0)

cars['Engine'] = cars['Engine'].astype(int)

print(cars['Engine'].unique())

cars.describe()

"""## Missing Value"""

zero_value = (cars == 0).sum()
zero_value

"""Beberapa kolom seperti Kilometres, Engine, City, dan Highway memiliki nilai 0, yang sebenarnya wajar. Nilai 0 pada Kilometres bisa menunjukkan mobil baru yang belum terpakai. Pada Engine, 0 mungkin merujuk pada kendaraan listrik atau yang tidak memiliki mesin berbahan bakar fosil. Sementara pada City dan Highway, nilai 0 bisa terjadi jika konsumsi bahan bakar belum diuji atau kendaraan baru yang efisien. Nilai 0 ini tidak mempengaruhi validitas data."""

cars[cars['Kilometres'] == 0]

cars.isna().sum()

"""## Data Duplicate"""

duplicate_rows = cars[cars.duplicated()]

if not duplicate_rows.empty:
    print("Duplicate Rows:")
    print(duplicate_rows)
else:
    print("No duplicate rows found.")

"""Terdapat duplikasi data pada dataset ini, di mana beberapa entri kendaraan memiliki informasi yang sama persis, termasuk Year, Make, Model, Kilometres, Engine, Transmission, Drivetrain, Exterior_Colour, Interior_Colour, Passengers, Doors, Fuel_Type, City, Highway, dan Price. Duplikasi ini terjadi pada beberapa baris berturut-turut dengan ID yang berbeda, yang kemungkinan disebabkan oleh kesalahan dalam pengumpulan atau pengolahan data. Data duplikat ini akan dihapus untuk mencegah pengaruh negatif pada pemodelan."""

cars = cars.drop_duplicates()

cars.shape

"""## Outliers"""

rows, cols = 6, 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 30))

# Iterasi untuk membuat boxplot pada setiap kolom
for i, column in enumerate(cars.columns):
    row, col = divmod(i, cols)
    sns.boxplot(ax=axes[row, col], x=cars[column])
    axes[row, col].set_title(f'Box Plot of {column}')

# Menghapus subplot yang tidak terpakai jika jumlah kolom lebih sedikit daripada jumlah subplot
for j in range(len(cars.columns), rows * cols):
    fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.show()

"""Pada boxplot ini, terlihat banyak outlier pada beberapa kolom, seperti Kilometres, Engine, City, Highway, dan Price. Nilai-nilai ekstrem ini berada jauh di luar rentang normal dan dapat mempengaruhi hasil analisis. Digunakan metode IQR untuk mendeteksi dan menangani outlier pada dataset.


"""

cars_numerik = cars.select_dtypes(include=['number'])

Q1 = cars_numerik.quantile(0.25)
Q3 = cars_numerik.quantile(0.75)
IQR=Q3-Q1
cars=cars[~((cars_numerik<(Q1-1.5*IQR))|(cars_numerik>(Q3+1.5*IQR))).any(axis=1)]

# Cek ukuran dataset setelah kita drop outliers
cars.shape

"""## Univariate Analysis"""

numerical_features = [
    'Year', 'Kilometres', 'Passengers', 'Doors', 'Engine',
    'City', 'Highway', 'Price'
]


categorical_features = [
    'Make', 'Model', 'Body_Type',
    'Transmission', 'Drivetrain', 'Exterior_Colour',
    'Interior_Colour', 'Fuel_Type'
]

"""### Categorical Features"""

feature = categorical_features[0]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Insight:

- Hyundai, Kia, Ford, dan Nissan merupakan merek yang paling banyak terwakili dalam dataset, menunjukkan popularitas dan potensi pasar yang kuat.
- Suzuki paling sedikit, mengindikasikan kurang populer.
- Meskipun terdapat ketidakseimbangan jumlah sampel antar merek, hal ini mencerminkan kondisi pasar mobil bekas yang sebenarnya dan dapat diatasi dengan teknik pemrosesan data yang sesuai.
- Keberagaman merek yang ada memungkinkan eksplorasi faktor-faktor yang mempengaruhi harga mobil bekas dari berbagai perspektif.

"""

feature = categorical_features[1]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Insight:

- Distribusi jumlah sampel per model memiliki pola sangat asimetris, dengan beberapa model menunjukkan volume yang sangat besar dibandingkan model lain.
- Civic memiliki jumlah sampel tertinggi, jauh melebihi model lain. Ini mengindikasikan bahwa Civic adalah model yang paling banyak dijual atau paling banyak terdapat dalam dataset.
- Sebagian besar model lainnya memiliki jumlah sampel yang relatif kecil, mengindikasikan model-model tersebut tidak terlalu banyak dijual atau terdapat dalam dataset.
"""

feature = categorical_features[2]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Insight:

- SUV merupakan jenis bodi yang paling umum, dengan lebih dari 5.000 sampel. Hal ini menunjukkan bahwa SUV adalah jenis kendaraan bekas yang paling banyak terdapat dalam dataset.
- Beberapa jenis bodi memiliki jumlah sampel yang relatif rendah, seperti Truck Long Crew Cab, Truck Short Crew Cab, dan Truck Long Crew Cab. Ini mengindikasikan bahwa jenis-jenis bodi tersebut kurang umum di pasar kendaraan bekas yang tercakup dalam dataset ini.
- Distribusi jenis bodi tidak merata, dengan beberapa jenis bodi mendominasi dataset, sementara banyak lainnya memiliki representasi yang jauh lebih sedikit.
"""

feature = categorical_features[3]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Insight:
- Automatic transmisi memiliki jumlah sampel yang jauh lebih besar dibandingkan jenis transmisi lainnya, menunjukkan bahwa transmisi otomatis sangat dominan dalam dataset kendaraan bekas ini.
- Manual transmisi memiliki jumlah sampel yang jauh lebih rendah dibandingkan automatic, mengindikasikan bahwa manual transmisi kurang populer di pasar kendaraan bekas yang tercakup dalam dataset.
- Terdapat beberapa varian automatic transmisi, seperti 5-Speed Automatic, 6-Speed Automatic, dan 10-Speed Automatic, masing-masing memiliki representasi yang cukup signifikan. Ini menunjukkan adanya diversifikasi jenis transmisi otomatis yang digunakan dalam kendaraan bekas.
- Untuk jenis manual transmisi, 5-Speed Manual dan 6-Speed Manual memiliki jumlah sampel yang paling besar, sementara varian lain seperti 8-Speed Manual dan 1-Speed Auto-Shift Manual memiliki representasi yang sangat terbatas.

"""

feature = categorical_features[4]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Insight:
- AWD (All-Wheel Drive) memiliki jumlah sampel yang jauh lebih besar dibandingkan jenis drivetrain lainnya, menunjukkan bahwa model kendaraan dengan penggerak roda empat (AWD) sangat dominan dalam dataset ini.
- FWD (Front-Wheel Drive) memiliki jumlah sampel yang cukup besar, meskipun jauh di bawah AWD. Hal ini mengindikasikan bahwa kendaraan dengan penggerak roda depan juga merupakan kategori yang umum dalam dataset.
- 4x4 (Four-Wheel Drive) memiliki jumlah sampel yang lebih rendah dibandingkan AWD dan FWD, namun masih cukup signifikan. Ini menunjukkan bahwa kendaraan dengan penggerak roda empat juga merupakan pilihan yang populer dalam pasar kendaraan bekas.
- RWD (Rear-Wheel Drive) dan 2WD (2-Wheel Drive) memiliki jumlah sampel yang jauh lebih sedikit dibandingkan tiga jenis drivetrain sebelumnya. Hal ini mengindikasikan bahwa model kendaraan dengan penggerak roda belakang atau dua roda tidak terlalu umum dalam dataset ini.
"""

feature = categorical_features[5]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Insight:
- Distribusi jenis warna eksterior kendaraan dalam dataset ini sangat beragam, dengan banyak kategori yang berbeda-beda.
- Warna eksterior yang paling dominan adalah Black, dengan jumlah sampel yang jauh lebih tinggi dibandingkan kategori warna lainnya. Ini menunjukkan bahwa kendaraan dengan warna gelap atau hitam merupakan pilihan populer dalam pasar kendaraan bekas.
- Selain Black, terdapat beberapa warna lain yang juga memiliki representasi cukup signifikan, seperti White, Gra. Ini mengindikasikan bahwa warna-warna netral atau solid juga diminati oleh konsumen kendaraan bekas.
- Sementara itu, terdapat banyak kategori warna dengan jumlah sampel yang sangat rendah, seperti Bronze, Turquoise, dan Lime Green. Hal ini mungkin menunjukkan bahwa warna-warna yang lebih jarang atau eksotis tidak sepopuler warna-warna standar.
"""

feature = categorical_features[6]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Insight:
- Berdasarkan grafik, warna interior yang paling dominan adalah Black dengan jumlah sampel yang jauh melebihi kategori warna interior lainnya. Ini menunjukkan bahwa kendaraan dengan interior berwarna gelap atau hitam sangat populer di pasar kendaraan bekas.
- Setelah Black, terdapat beberapa warna interior lain yang juga memiliki representasi yang cukup signifikan, seperti Gray, Tan, dan Brown. Hal ini mengindikasikan bahwa konsumen juga menyukai pilihan warna interior yang lebih netral atau alami.
- Sementara itu, beberapa kategori warna interior seperti Blue, Burgundy, dan Ivory memiliki jumlah sampel yang relatif rendah. Ini mungkin menandakan bahwa pilihan warna interior yang lebih berani atau unik kurang diminati dalam pasar kendaraan bekas.
- Secara keseluruhan, visualisasi fitur Interior_Colour menunjukkan preferensi konsumen yang jelas terhadap warna interior gelap atau netral.
"""

feature = categorical_features[7]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Insight:
- Dari grafik, dapat dilihat bahwa jenis bahan bakar yang paling dominan adalah Gas. Jumlah sampel untuk Gas sangat tinggi, jauh melebihi kategori bahan bakar lainnya.
- Setelah Gas, terdapat beberapa kategori bahan bakar lain yang juga memiliki representasi yang cukup signifikan, seperti Gasoline-Hybrid, Diesel, dan Flex-Fuel. Ini menunjukkan bahwa selain gas, ada pula preferensi konsumen terhadap opsi bahan bakar lainnya.
- Sementara itu, kategori bahan bakar seperti Electric, Other, dan Gasoline/Electric Hybrid memiliki jumlah sampel yang relatif rendah. Hal ini bisa berarti bahwa kendaraan berbahan bakar elektrik atau hibrida belum terlalu populer dalam dataset kendaraan bekas ini.
- Secara keseluruhan, visualisasi fitur Fuel_Type mengungkapkan bahwa kendaraan berbahan bakar gas adalah pilihan dominan di pasar kendaraan bekas

### Numerical Features
"""

cars.hist(bins=50, figsize=(20,15))
plt.show()

"""Insight:
- Tahun Pembuatan (Year):
Distribusi tahun pembuatan menunjukkan pola yang menarik, dengan peningkatan signifikan pada tahun 2016 hingga mencapai puncak pada tahun 2018 dan 2022. Hal ini menunjukkan bahwa dataset ini lebih fokus pada kendaraan bekas yang relatif baru, dengan data yang lebih banyak di tahun-tahun terkini hingga 2022. Pola ini memberikan wawasan tentang tren pasar kendaraan bekas yang semakin mengarah pada model-model terbaru.
- Kilometer (Kilometres):
Distribusi kilometer menunjukkan tren umum dalam pasar kendaraan bekas, dengan jumlah kendaraan terbanyak berada pada rentang kilometer yang lebih rendah. Hal ini mencerminkan preferensi konsumen terhadap kendaraan bekas dengan kondisi yang masih baik dan jarak tempuh yang tidak terlalu tinggi. Data ini dapat memberikan gambaran tentang segmen pasar kendaraan bekas yang lebih diminati, serta faktor-faktor yang mempengaruhi harga.
- Jumlah Mesin (Engine):
Sebagian besar kendaraan dalam dataset ini dilengkapi dengan mesin berkapasitas 4 atau 5 silinder. Ini menunjukkan bahwa mesin dengan kapasitas tersebut cenderung lebih umum dan diminati oleh konsumen karena memberikan keseimbangan antara performa dan efisiensi bahan bakar. Analisis lebih lanjut mengenai hubungan kapasitas mesin, efisiensi bahan bakar, dan harga kendaraan dapat memberikan wawasan berharga dalam memahami preferensi pasar.
- Jumlah Penumpang (Passengers) dan Pintu (Doors):
Sebagian besar kendaraan dalam dataset ini memiliki kapasitas 4 hingga 5 penumpang dan 4 hingga 5 pintu. Hal ini mengindikasikan bahwa kendaraan yang lebih banyak digunakan untuk keluarga atau tujuan multi-guna lebih dominan di pasar kendaraan bekas. Memahami hubungan antara jumlah penumpang, pintu, dan atribut lainnya dapat memberikan wawasan lebih lanjut mengenai preferensi konsumen dan pengaruhnya terhadap harga.
- Konsumsi Bahan Bakar (City, Highway):
Distribusi konsumsi bahan bakar menunjukkan bahwa sebagian besar kendaraan dalam dataset memiliki efisiensi bahan bakar yang cukup baik, dengan angka konsumsi yang lebih rendah, baik di kota maupun di jalan tol. Hal ini mencerminkan tren konsumen yang lebih memilih kendaraan hemat bahan bakar, terutama mengingat harga bahan bakar yang terus meningkat. Analisis lebih lanjut dapat mengidentifikasi hubungan antara efisiensi bahan bakar, harga kendaraan, dan faktor-faktor lainnya yang memengaruhi pasar.
- Harga (Price):
Distribusi harga kendaraan bekas menunjukkan variasi yang cukup besar, dari harga yang relatif rendah hingga menengah. Hal ini mencerminkan keragaman dalam kondisi, model, dan fitur kendaraan dalam dataset. Analisis hubungan antara harga dan fitur-fitur lainnya dapat membantu dalam merancang model prediksi harga yang lebih akurat, dengan mempertimbangkan berbagai faktor yang memengaruhi harga kendaraan bekas.

## Multivariate Analysis

### Categorical Feature
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Menentukan kolom kategorikal
cat_features = cars.select_dtypes(include='object').columns.to_list()

# Menentukan jumlah baris dan kolom secara dinamis
n_cols = 3  # Jumlah kolom dalam grid
n_rows = -(-len(cat_features) // n_cols)  # Hitung jumlah baris (ceiling division)

# Membuat grid subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()  # Meratakan grid untuk iterasi

# Membuat plot untuk setiap kolom kategorikal
for i, col in enumerate(cat_features):
    sns.barplot(
        x=col,
        y="Price",
        data=cars,
        ax=axes[i],
        palette="Set3",
        ci=None  # Interval kepercayaan dihilangkan untuk kejelasan
    )
    axes[i].set_title(f"Rata-rata 'Price' terhadap {col}", fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)  # Membuat label x miring
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Rata-rata Harga')

# Menghapus subplot kosong (jika ada)
for j in range(len(cat_features), len(axes)):
    fig.delaxes(axes[j])

# Menyesuaikan tata letak
plt.tight_layout()
plt.show()

"""Insight:
- Merek vs Harga: Grafik pertama menunjukkan variasi yang signifikan pada harga rata-rata kendaraan bekas di antara berbagai merek, dengan beberapa merek seperti Toyota dan Honda memiliki harga yang jauh lebih tinggi dibandingkan merek lain seperti Mitsubishi dan Geo. Hal ini mengindikasikan bahwa merek kendaraan merupakan faktor penting yang menentukan harga kendaraan bekas.
- Model vs Harga: Grafik kedua semakin memperkuat bahwa model kendaraan juga merupakan faktor kritis, dengan variabilitas harga rata-rata yang bahkan lebih besar di antara model-model yang berbeda dalam satu merek yang sama. Beberapa model tampaknya mampu mempertahankan nilai jualnya jauh lebih baik dibandingkan model lain.
- Jenis Bodi vs Harga: Grafik ketiga menunjukkan bahwa jenis bodi kendaraan juga merupakan variabel kunci, dengan SUV, truk, dan beberapa sedan mewah cenderung berada pada kisaran harga yang lebih tinggi, sementara sedan dan hatchback dasar umumnya memiliki harga yang lebih rendah.
- Transmisi vs Harga: Grafik keempat memperlihatkan bahwa jenis transmisi juga berperan, dengan transmisi otomatis memiliki harga rata-rata yang lebih tinggi dibandingkan manual.
- Drivetrain vs Harga: Grafik kelima mengindikasikan bahwa konfigurasi Drivetrain all-wheel drive dan four-wheel drive terkait dengan harga kendaraan bekas yang lebih tinggi dibandingkan penggerak roda depan atau belakang.
- Warna Eksterior vs Harga: Grafik keenam mengungkapkan bahwa warna eksterior juga dapat memengaruhi harga, dengan warna netral seperti putih, hitam, dan abu-abu cenderung memiliki harga rata-rata yang lebih tinggi dibandingkan warna yang lebih berani atau tidak biasa.
- Warna Interior vs Harga: Serupa dengan itu, grafik ketujuh menunjukkan bahwa warna interior juga berpengaruh terhadap harga kendaraan bekas, dengan interior berwarna hitam menjadi yang paling bernilai secara rata-rata.
- Jenis Bahan Bakar vs Harga: Grafik kedelapan memperlihatkan bahwa kendaraan berbahan bakar bensin umumnya memiliki harga kendaraan bekas yang paling tinggi, sementara tipe bahan bakar alternatif seperti listrik dan hibrida cenderung memiliki harga yang lebih rendah.

### Numerical Features
"""

sns.pairplot(cars, diag_kind = 'kde')

print(cars[['Doors', 'Passengers']].describe())

"""Berdasarkan deskripsi statistik dari kedua fitur ini, yaitu Doors dan Passengers, dapat dilihat bahwa keduanya memiliki nilai yang sama secara konsisten untuk semua baris data yaitu Doors selalu bernilai 4, dan Passengers selalu bernilai 5.

Hal ini menunjukkan bahwa kedua fitur ini tidak mengandung variasi atau informasi yang berguna karena nilainya statis di seluruh dataset. Ketika diperiksa lebih lanjut dalam correlation matrix, keduanya tidak memiliki hubungan yang berarti dengan fitur lainnya, bahkan tidak ada nilai numerik yang muncul. Oleh karena itu, kedua fitur ini harus di-drop, karena mereka tidak memberikan kontribusi apa pun terhadap model dan hanya akan menambah kompleksitas data tanpa menambah informasi yang berguna
"""

cars = cars.drop(['Doors', 'Passengers'], axis=1)
cars.head()

numerical_features = [col for col in numerical_features if col in cars.columns]

plt.figure(figsize=(10, 8))
correlation_matrix = cars[numerical_features].corr(method='spearman').round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix untuk Fitur Numerik", size=20)
plt.show()

"""Insight:
- Tahun (Year) vs Lainnya
  
  Tahun menunjukkan korelasi negatif yang kuat dengan Kilometer (-0,89) dan Harga (-0,77). Ini berarti semakin tua kendaraan, semakin rendah jarak tempuh dan harga jual kendaraan bekas.

- Kilometer (Kilometres) vs Lainnya
  
  Kilometer memiliki korelasi negatif dengan Tahun (-0,89) dan Harga (-0,34). Dengan kata lain, kendaraan dengan jarak tempuh lebih tinggi cenderung lebih tua dan memiliki harga yang lebih rendah.

- Mesin (Engine) vs Lainnya
  
  Kapasitas mesin menunjukkan korelasi positif sedang dengan konsumsi bahan bakar di Kota (0,52) dan Jalan Tol (0,63). Artinya, mesin berkapasitas lebih besar biasanya dihubungkan dengan konsumsi bahan bakar yang lebih tinggi.

- Kota (City) vs Lainnya
  
  Konsumsi bahan bakar di Kota memiliki korelasi positif sedang dengan Mesin (0,52) dan sangat kuat dengan Jalan Tol (0,91). Ini menunjukkan hubungan erat antara konsumsi bahan bakar di kota dan jalan tol, serta pengaruh kapasitas mesin terhadap keduanya.

- Jalan Tol (Highway) vs Lainnya
  
  Jalan Tol memiliki korelasi sangat kuat dengan Kota (0,91) dan sedang dengan Mesin (0,63). Hal ini menguatkan bahwa konsumsi bahan bakar di kota dan jalan tol saling terkait erat, dan kapasitas mesin turut berkontribusi pada pola konsumsi tersebut.

#Data Preparation

##Encode Fitur Kategori
"""

from sklearn.preprocessing import  OneHotEncoder
cars = pd.concat([cars, pd.get_dummies(cars['Make'], prefix='Make')],axis=1)
cars = pd.concat([cars, pd.get_dummies(cars['Model'], prefix='Model')],axis=1)
cars = pd.concat([cars, pd.get_dummies(cars['Body_Type'], prefix='Body_Type')],axis=1)
cars = pd.concat([cars, pd.get_dummies(cars['Transmission'], prefix='Transmission')],axis=1)
cars = pd.concat([cars, pd.get_dummies(cars['Drivetrain'], prefix='Drivetrain')],axis=1)
cars = pd.concat([cars, pd.get_dummies(cars['Exterior_Colour'], prefix='Exterior_Colour')],axis=1)
cars = pd.concat([cars, pd.get_dummies(cars['Interior_Colour'], prefix='Interior_Colour')],axis=1)
cars = pd.concat([cars, pd.get_dummies(cars['Fuel_Type'], prefix='Fuel_Type')],axis=1)
cars.drop(['Make', 'Model', 'Body_Type', 'Transmission', 'Drivetrain', 'Exterior_Colour', 'Interior_Colour', 'Fuel_Type'], axis=1, inplace=True)

cars.head()

"""##PCA"""

sns.pairplot(cars[['City','Highway']], plot_kws={"s": 2});

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=123)
pca.fit(cars[['City','Highway']])
princ_comp = pca.transform(cars[['City','Highway']])

pca.explained_variance_ratio_.round(2)

from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=123)
pca.fit(cars[['City','Highway']])
cars['Efficiency'] = pca.transform(cars.loc[:, ('City','Highway')]).flatten()
cars.drop(['City','Highway'], axis=1, inplace=True)

cars.head()

"""## Train Test Split"""

from sklearn.model_selection import train_test_split

X = cars.drop(["Price"],axis =1)
y = cars["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""## Standarisasi"""

from sklearn.preprocessing import StandardScaler

numerical_features = ['Year', 'Kilometres', 'Efficiency', 'Engine']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(3)

"""# Model Development"""

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""## KNN Model"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=13)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""## Random Forest Model"""

# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor

# buat model prediksi
RF = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""## Ada Boosting Model

"""

from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(n_estimators=250,
                             learning_rate=0.01,
                             random_state=50)

boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""# Evaluasi"""

X_test[numerical_features] = scaler.transform(X_test.loc[:, numerical_features])
X_test[numerical_features].head()

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

# Panggil mse
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[:10].copy()
pred_dict = {'y_true':y_test[:10]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)

gridtuning = pd.DataFrame(index=['KNN', 'RF', 'Adaboost'], columns=['train_mse', 'test_mse'])

from sklearn.model_selection import GridSearchCV

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15]
}

knn = KNeighborsRegressor()

grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_knn.fit(X_train, y_train)

gridtuning.loc['KNN', 'train_mse'] = mean_squared_error(y_pred=grid_search_knn.predict(X_train), y_true=y_train)
gridtuning.loc['KNN', 'test_mse'] = mean_squared_error(y_pred=grid_search_knn.predict(X_test), y_true=y_test)

param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=35, n_jobs=-1)

grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

gridtuning.loc['RF', 'train_mse'] = mean_squared_error(y_pred=grid_search_rf.predict(X_train), y_true=y_train)
gridtuning.loc['RF', 'test_mse'] = mean_squared_error(y_pred=grid_search_rf.predict(X_test), y_true=y_test)

param_grid_ada = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 1.0]
}

ada = AdaBoostRegressor(random_state=42)

grid_search_ada = GridSearchCV(ada, param_grid_ada, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_ada.fit(X_train, y_train)

gridtuning.loc['Adaboost', 'train_mse'] = mean_squared_error(y_pred=grid_search_ada.predict(X_train), y_true=y_train)
gridtuning.loc['Adaboost', 'test_mse'] = mean_squared_error(y_pred=grid_search_ada.predict(X_test), y_true=y_test)

gridtuning

fig, ax = plt.subplots()
gridtuning.sort_values(by='test_mse', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[:10].copy()
pred_dict = {'y_true':y_test[:10]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
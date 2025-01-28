# Laporan Proyek Machine Learning - Febrian Chrisna Ardianto

## Domain Proyek

Industri otomotif, khususnya pasar mobil bekas, telah berkembang pesat dalam beberapa dekade terakhir, seiring dengan meningkatnya kebutuhan masyarakat akan kendaraan yang terjangkau dan berkualitas. Namun, pasar mobil bekas memiliki tantangan tersendiri, terutama dalam hal penentuan harga yang akurat. Penjual sering kali menetapkan harga berdasarkan pengalaman subjektif atau perbandingan sederhana dengan mobil serupa, sementara pembeli mengandalkan negosiasi tanpa memahami nilai sebenarnya dari kendaraan yang mereka beli. Akibatnya, sering terjadi ketidakseimbangan antara harga pasar dengan nilai kendaraan, yang dapat merugikan baik penjual maupun pembeli.

Saat ini industri mobil bekas menghadapi tantangan besar dalam strategi penetapan harga yang dapat mengakibatkan kerugian finansial yang signifikan. Menurut analisis yang dilakukan oleh Indicata UK, dealer mobil bekas di Inggris diproyeksikan mengalami kerugian hampir £500 juta pada tahun 2024 akibat kesalahan dalam menentukan harga kendaraan. Dua kesalahan utama yang teridentifikasi dalam studi ini adalah menetapkan harga terlalu tinggi untuk kendaraan yang lambat terjual dan menjual kendaraan populer terlalu cepat dengan harga pasar tanpa mempertimbangkan premi yang dapat diperoleh.

Untuk mengatasi permasalahan ini, analisis data prediktif berbasis machine learning dapat digunakan untuk mengoptimalkan strategi harga mobil bekas. Model prediktif dapat membantu mengidentifikasi faktor-faktor utama yang mempengaruhi harga kendaraan. Dengan pendekatan ini, penjual dapat menghindari undervaluation pada kendaraan dengan permintaan tinggi dan menyesuaikan harga kendaraan yang kurang diminati secara lebih efektif.
  
Referensi: [Mispricing of used cars could cost UK dealers nearly £500m in 2024](https://www.motorfinanceonline.com/news/indicata-uk-used-car-mispricing/?cf-view) 

## Business Understanding

### Problem Statements

- Bagaimana faktor-faktor seperti tahun pembuatan mobil, jarak tempuh, konsumsi bahan bakar dalam kota (City) dan luar kota (Highway), serta kondisi kendaraan memengaruhi harga mobil bekas?
- Model machine learning apa yang dapat memberikan prediksi terbaik untuk menentukan harga mobil bekas dengan akurasi tinggi?

### Goals

- Menganalisis bagaimana variabel seperti tahun pembuatan mobil, jarak tempuh, konsumsi bahan bakar dalam kota (City) dan luar kota (Highway) memengaruhi harga mobil bekas.
- Mengidentifikasi algoritma machine learning yang dapat memberikan akurasi terbaik untuk memprediksi harga mobil bekas.

### Solution statements
- Menggunakan algoritma dasar seperti K-Nearest Neighbors (KNN) dan Random Forest sebagai model awal untuk memprediksi harga mobil bekas.
- Menggunakan algoritma lanjutan seperti Ada Boost Regressor
- Melakukan hyperparameter tuning menggunakan teknik seperti Grid Search untuk menemukan kombinasi parameter yang optimal dan meningkatkan performa model.
- Mengevaluasi model dengan menggunakan metrik seperti MSE untuk mendapatkan gambaran yang lebih baik mengenai seberapa baik model memprediksi harga mobil bekas.


## Data Understanding
Dataset ini berisi informasi mengenai 24.199 kendaraan bekas yang terdaftar untuk dijual pada tahun 2023. Kendaraan-kendaraan tersebut berada dalam radius 25 kilometer dari pusat kota Toronto, Ontario, Kanada, dan datanya diambil dari platform Autotrader.ca. Dengan skor Usability sebesar 10.00, dataset ini telah melalui evaluasi yang ketat dan dinilai  layak untuk mendukung penelitian machine learning. Penilaian ini mencakup aspek kemudahan penggunaan, efisiensi, serta kualitas data yang memenuhi standar tinggi dari kaggle.

Referensi: [Used Vehicles For Sale](https://www.kaggle.com/datasets/farhanhossein/used-vehicles-for-sale?).

### Deskripsi Variable
|    Nama Kolom     |                Deskripsi                       |
|-------------------|------------------------------------------------|
| Year              | Tahun pembuatan mobil (numerik)                |
| Make              | Merek kendaraan (kategorikal)                  |
| Model             | Model kendaraan (kategorikal)                  |
| Kilometres        | Jumlah kilometer yang telah ditempuh (numerik) |
| Body_Type         | Jenis tubuh kendaraan (kategorikal)            |   
| Engine            | Jenis mesin kendaraan (kategorikal)            |
| Transmission      | Jenis transmisi kendaraan (kategorikal)        |
| Drivetrain        | Jenis penggerak kendaraan (kategorikal)        |
| Exterior_Colour   | Warna eksterior kendaraan (kategorikal)        |
| Interior_Colour   | Warna interior kendaraan (kategorikal)         |
| Passengers        | Jumlah penumpang kendaraan (numerik)           |
| Doors             | Jumlah pintu kendaraan (numerik)               |
| Fuel_Type         | Jenis bahan bakar kendaraan (kategorikal)      |
| City              | Konsumsi bahan bakar di kota (numerik)         |
| Highway           | Konsumsi bahan bakar di jalan tol (numerik)    |
| Price             | Harga kendaraan (numerik)                      |


## Exploratory Data Analysis

### Kesalahan Tipe Data
![Missing Value](assets/kesalahan_tipedata.png)

Pada kolom Engine, terdapat kesalahan tipe data dimana nilai string 'E' muncul, padahal seharusnya nilai tersebut adalah 0, yang menunjukkan bahwa mobil tersebut tidak memiliki mesin berbahan bakar fosil atau merupakan mobil listrik. Untuk memperbaiki hal ini, nilai 'E' telah diganti dengan 0, dan tipe data kolom tersebut telah diubah menjadi integer agar lebih sesuai dengan format yang diinginkan dan dapat digunakan dalam analisis lebih lanjut.

### Missing Value
Dataset yang digunakan tidak memiliki data null atau kosong. Namun dalam dataset terdapat baris data dengan value 0.

![Missing Value](assets/missing_value.png)

Beberapa kolom seperti Kilometres, Engine, City, dan Highway memiliki nilai 0, yang sebenarnya wajar. Nilai 0 pada Kilometres bisa menunjukkan mobil baru yang belum terpakai. Pada Engine, 0 mungkin merujuk pada kendaraan listrik atau yang tidak memiliki mesin berbahan bakar fosil. Sementara pada City dan Highway, nilai 0 bisa terjadi jika konsumsi bahan bakar belum diuji atau kendaraan baru yang efisien. Nilai 0 ini tidak mempengaruhi validitas data.

Dataset ini awalnya memiliki beberapa baris duplikat, yang dapat terjadi karena kesalahan pencatatan atau data yang dimasukkan lebih dari sekali. Namun, untuk memastikan kualitas dan akurasi data, baris-baris duplikat tersebut telah diidentifikasi dan dihapus, sehingga dataset ini kini hanya berisi entri yang unik dan valid.

### Menangani Outlier
![Outliers](assets/outliers.png)

Dataset ini mengandung outliers yang terdeteksi pada beberapa kolom tertentu. Dalam proyek ini, digunakan metode IQR untuk mendeteksi dan menangani outlier pada dataset.

Mengapa menggunakan IQR?
- IQR adalah metode statistik yang sederhana dan tidak bergantung pada asumsi distribusi data.
- Metode ini menghitung rentang antar kuartil (Q3 - Q1), di mana Q1 adalah kuartil pertama (25%) dan Q3 adalah kuartil ketiga (75%).
- Nilai-nilai yang berada di luar rentang [Q1 - 1.5IQR, Q3 + 1.5IQR] dianggap sebagai outlier.
- Cocok untuk data dengan distribusi tidak normal dan lebih tahan terhadap pengaruh outlier ekstrem dibandingkan metode lain.


```python
cars_numerik = cars.select_dtypes(include=['number'])```

Q1 = cars_numerik.quantile(0.25) ```
Q3 = cars_numerik.quantile(0.75)
IQR=Q3-Q1
cars=cars[~((cars_numerik<(Q1-1.5*IQR))|(cars_numerik>(Q3+1.5*IQR))).any(axis=1)]

# Cek ukuran dataset setelah kita drop outliers
cars.shape
```

### Univariate Analysis
#### Categorical Features
Make:
Model:
Body_Type:
Transmission:
Drivetrain:
Exterior_Colour:
Interior_Colour:
Fuel_Type:

### Numerical Features

## Multivariate Analysis
### Categorical Features

### Correlation Matrix

## Data Preparation
Proses data preparation diperlukan untuk memastikan data yang bersih dan siap digunakan oleh model machine learning. Data preparation meliputi langkah-langkah berikut:
1. Encoding Fitur Kategori: Fitur kategori seperti Make, Model, Body_Type, dll., diubah menjadi format numerik menggunakan One-Hot Encoding. Tujuan dari encoding ini adalah agar model machine learning dapat memproses data kategorikal dengan lebih baik karena sebagian besar model hanya bisa bekerja dengan data numerik
2. Train Test Split: Dataset dibagi menjadi 80% data pelatihan (X_train, y_train) dan 20% data pengujian (X_test, y_test). Fitur (X) adalah semua kolom kecuali Price, sementara target (y) adalah kolom Price
3. Standarisasi: Fitur numerik seperti Year, Kilometres, Efficiency, dan Engine distandarisasi menggunakan StandardScaler. Tujuan standarisasi adalah untuk membawa semua fitur ke skala yang sama, sehingga model tidak akan condong ke fitur dengan nilai lebih besar, menghindari bias dalam pemodelan
4. Reduksi dimensi fitur dengan PCA: Fitur City dan Highway digabungkan menjadi satu fitur baru Efficiency menggunakan PCA karena kedua fitur tersebut memiliki korelasi yang tinggi. Tujuan dari PCA adalah untuk mengurangi dimensi fitur, mengurangi kompleksitas data, dan menangkap informasi penting dalam satu komponen, sehingga model menjadi lebih efisien

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Algoritma yang digunakan yaitu:
1. **K-Nearest Neighbor**

    K-Nearest Neighbor (KNN) adalah algoritma supervised learning yang digunakan untuk regresi dan klasifikasi. Prinsip dasar dari KNN adalah bahwa prediksi untuk suatu data baru dilakukan berdasarkan mayoritas nilai atau rata-rata dari k tetangga terdekatnya di ruang fitur. KNN tidak memerlukan model eksplisit dan memanfaatkan jarak antar data untuk membuat keputusan. Dalam regresi, nilai prediksi dihitung sebagai rata-rata nilai target dari k tetangga terdekat.

    **Parameter:**
    - `n_neighbors=13:` Menentukan jumlah tetangga terdekat yang akan digunakan untuk menghitung prediksi. Semakin besar nilai k, semakin smooth modelnya, tetapi terlalu besar bisa mengurangi akurasi jika data tidak linier.
      
  ```python
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    
    knn = KNeighborsRegressor(n_neighbors=13)
    knn.fit(X_train, y_train)
    
    models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
  ```
2. Random Forest

    Random Forest adalah algoritma ensemble learning yang menggabungkan beberapa pohon keputusan (decision trees). Setiap pohon keputusan dibuat dengan subset acak dari data dan fitur, dan hasilnya digabungkan untuk memberikan prediksi akhir. Random Forest mengurangi overfitting yang sering terjadi pada pohon keputusan tunggal dengan cara menggabungkan banyak model.

    Parameter:
    - `n_estimators=100:` Menentukan jumlah pohon keputusan dalam hutan. Semakin banyak pohon, semakin kuat modelnya, meskipun dengan biaya komputasi yang lebih tinggi.
    - `max_depth=10:` Menentukan kedalaman maksimum setiap pohon keputusan. Membatasi kedalaman pohon dapat membantu menghindari overfitting.
    - `random_state=55:` Mengatur seed untuk menghasilkan hasil yang dapat diulang.
    - `n_jobs=-1:` Menggunakan semua core prosesor untuk mempercepat komputasi.
   
  ```python
    from sklearn.ensemble import RandomForestRegressor
    
    # buat model prediksi
    RF = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=55, n_jobs=-1)
    RF.fit(X_train, y_train)
    
    models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
   ```
2. Adaptive Boosting

    AdaBoost (Adaptive Boosting) adalah algoritma ensemble yang membangun model prediktif secara iteratif. Pada setiap iterasi, model yang lebih lemah diberikan bobot lebih besar untuk memperbaiki kesalahan yang dibuat oleh model sebelumnya. AdaBoost bekerja dengan meningkatkan model-model yang lemah menjadi model yang lebih kuat dengan memberikan perhatian lebih pada data yang sulit diprediksi

    Parameter:
    - `n_estimators=100:` Menentukan jumlah iterasi (jumlah model yang akan digabungkan). Semakin besar jumlah estimators, semakin kuat modelnya.
    - `random_state=55:` Mengatur seed untuk menghasilkan hasil yang dapat diulang.
      
  ```python
    from sklearn.ensemble import AdaBoostRegressor
    
    boosting = AdaBoostRegressor(n_estimators=250,
                                 learning_rate=0.01,
                                 random_state=50)
    
    boosting.fit(X_train, y_train)
    models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
  ```

## Evaluation
### Metrik Evaluasi

Mean Squared Error (MSE) digunakan sebagai metrik evaluasi untuk model regresi. MSE menghitung rata-rata kuadrat dari selisih antara nilai aktual yi dan nilai prediksi ŷi. Metrik ini sangat sensitif terhadap error besar karena perhitungan kuadratnya.

**Rumus:**
![Missing Value](assets/diagram.png)

**Penjelasan:**
- N = Jumlah data dalam dataset.
- yi = Nilai target aktual
- y_pred = Nilai prediksi model

MSE yang lebih kecil menunjukkan bahwa model memiliki prediksi yang lebih akurat terhadap nilai target.

### Hasil Sebelum Hyperparameter Tuning
![Missing Value](assets/diagram.png)

### Hasi Setelah Hyperparameter Tuning

## Kesimpulan

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.


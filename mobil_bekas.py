# -*- coding: utf-8 -*-
"""mobil_bekas_fix.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1O5uQeJzD-VoUyR4KXXXur4m77hbhiif1

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

"""## Kesalahan Tipe Data"""

cars['Engine'] = cars['Engine'].replace('E', 0)

cars['Engine'] = cars['Engine'].astype(int)

print(cars['Engine'].unique())

cars.describe()

"""## Missing Value"""

zero_value = (cars == 0).sum()
zero_value

cars[cars['Kilometres'] == 0]

cars.isna().sum()

"""## Data Duplicate"""

duplicate_rows = cars[cars.duplicated()]

if not duplicate_rows.empty:
    print("Duplicate Rows:")
    print(duplicate_rows)
else:
    print("No duplicate rows found.")

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

# Pilih fitur kategori
feature = categorical_features[1]

# Hitung jumlah dan persentase kategori
count = cars[feature].value_counts()
percent = 100 * cars[feature].value_counts(normalize=True)

# Ambil 10 kategori teratas berdasarkan jumlah sampel
top_10_count = count.nlargest(10)
top_10_percent = percent.loc[top_10_count.index]

# Buat DataFrame untuk 10 kategori teratas
df_top_10 = pd.DataFrame({'jumlah sampel': top_10_count, 'persentase': top_10_percent.round(1)})
print(df_top_10)

# Plot bar chart untuk 10 kategori teratas
top_10_count.plot(kind='bar', title=f"{feature}", figsize=(8, 6))
plt.ylabel('Jumlah Sampel')
plt.xlabel(feature)
plt.xticks(rotation=45)
plt.show()

feature = categorical_features[2]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[3]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[4]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

# Pilih fitur kategori
feature = categorical_features[5]

# Hitung jumlah dan persentase kategori
count = cars[feature].value_counts()
percent = 100 * cars[feature].value_counts(normalize=True)

# Ambil 10 kategori teratas berdasarkan jumlah sampel
top_10_count = count.nlargest(10)
top_10_percent = percent.loc[top_10_count.index]

# Buat DataFrame untuk visualisasi
df_top_10 = pd.DataFrame({'jumlah sampel': top_10_count, 'persentase': top_10_percent.round(1)})
print(df_top_10)

# Plot bar chart untuk 10 kategori teratas
top_10_count.plot(kind='bar', title=f"{feature}", figsize=(8, 6))
plt.ylabel('Jumlah Sampel')
plt.xlabel(feature)
plt.xticks(rotation=45)
plt.show()

feature = categorical_features[6]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[7]
count = cars[feature].value_counts()
percent = 100*cars[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""### Numerical Features"""

cars.hist(bins=50, figsize=(20,15))
plt.show()

"""## Multivariate Analysis

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

top_n = 10  # Menampilkan 10 kategori teratas

for i, col in enumerate(cat_features):
    top_categories = cars.groupby(col)['Price'].mean().nlargest(top_n).index  # Top-N berdasarkan rata-rata harga
    filtered_data = cars[cars[col].isin(top_categories)]  # Filter data hanya untuk top categories

    sns.barplot(
        x=col,
        y="Price",
        data=filtered_data,
        ax=axes[i],
        palette="Set3",
        ci=None
    )
    axes[i].set_title(f"Rata-rata 'Price' terhadap {col} (Top {top_n})", fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Rata-rata Harga')


# Menghapus subplot kosong (jika ada)
for j in range(len(cat_features), len(axes)):
    fig.delaxes(axes[j])

# Menyesuaikan tata letak
plt.tight_layout()
plt.show()

"""### Numerical Features"""

sns.pairplot(cars, diag_kind = 'kde')

print(cars[['Doors', 'Passengers']].describe())

cars = cars.drop(['Doors', 'Passengers'], axis=1)
cars.head()

numerical_features = [col for col in numerical_features if col in cars.columns]

plt.figure(figsize=(10, 8))
correlation_matrix = cars[numerical_features].corr(method='spearman').round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix untuk Fitur Numerik", size=20)
plt.show()

"""#Data Preparation

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
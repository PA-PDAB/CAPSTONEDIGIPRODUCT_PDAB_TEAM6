import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class KNeighborsClassifier:
  def __init__(self, n_neighbors=5):
      # Inisialisasi jumlah tetangga yang akan digunakan dalam klasifikasi
      self.n_neighbors = n_neighbors

  def fit(self, X, y):
      # Menyimpan data latih dalam bentuk array NumPy
      self.X_train = np.array(X)
      self.y_train = np.array(y)

  def predict(self, X_test):
      # Mengkonversi data uji ke dalam bentuk array NumPy
      X_test = np.array(X_test)
      # Inisialisasi array kosong untuk menyimpan prediksi kelas
      y_pred = np.zeros(X_test.shape[0], dtype=self.y_train.dtype)
      # Loop melalui setiap sampel data uji
      for i, x_test in enumerate(X_test):
          # Menghitung jarak antara sampel uji dengan setiap sampel data latih
          distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
          # Mengurutkan indeks berdasarkan jarak dan mengambil indeks dari n tetangga terdekat
          indices = np.argsort(distances)[:self.n_neighbors]
          # Mengambil kelas dari n tetangga terdekat
          k_nearest_classes = self.y_train[indices]
          # Menggunakan majority voting untuk menentukan kelas prediksi
          y_pred[i] = np.argmax(np.bincount(k_nearest_classes))
      return y_pred
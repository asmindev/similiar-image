# Dokumentasi Pencarian Kesamaan Gambar

Dokumentasi ini memberikan gambaran tentang program pencarian kesamaan gambar yang menggunakan Flask sebagai antarmuka pengguna dan model Convolutional Neural Network (CNN) dengan model Xception untuk ekstraksi fitur gambar.

## Pendahuluan

Program pencarian kesamaan gambar ini dirancang untuk memungkinkan pengguna mencari gambar yang mirip berdasarkan gambar permintaan. Program ini menggunakan model Xception untuk ekstraksi fitur dan memanfaatkan Flask untuk menyediakan antarmuka yang ramah pengguna untuk berinteraksi dengan fungsionalitas pencarian.

## Persyaratan

Teknologi dan pustaka berikut digunakan dalam program ini:
- Python
- Flask
- OpenCV (cv2)
- Numpy
- Scikit-learn (NearestNeighbors)
- Keras (Xception)
- Matplotlib (untuk visualisasi)
- Joblib

## Struktur Program

Program ini terbagi menjadi dua bagian utama: aplikasi Flask dan modul pencarian gambar (`v2`).

### Aplikasi Flask

1. **Rute Indeks (`/`):** Menampilkan daftar gambar dari dataset yang diacak di halaman utama.

2. **Rute Anjing (`/dog/<string:dog_name>`):** Menampilkan detail gambar anjing tertentu ketika namanya diberikan dalam URL.

3. **Rute Kesamaan (`/similarity`):** Menerima permintaan POST JSON dengan nama gambar permintaan, mencari gambar yang mirip, dan mengembalikan hasil dalam bentuk JSON.

### Modul Pencarian Gambar (`v2`)

Kelas `ImageSearch` menangani ekstraksi fitur gambar dan pencarian kesamaan menggunakan model Xception.

1. **Inisialisasi:** Kelas diinisialisasi dengan direktori yang berisi gambar dataset, jalur untuk model Xception, dan berkas vektor fitur.

2. **Ekstraksi Fitur:** Metode `extract_features` memproses gambar dataset melalui model Xception dan mengekstrak vektor fiturnya.

3. **Penyimpanan/Pemuatan Fitur:** Vektor fitur disimpan menggunakan Joblib untuk menghindari perhitungan ulang dan meningkatkan efisiensi pencarian.

4. **Pencarian Gambar:** Metode `search_similar_images` mengambil jalur gambar permintaan dan mengembalikan indeks gambar yang mirip dengan menggunakan jarak kosinus.

5. **Pencarian dan Dapatkan Hasil:** Metode `search_and_get_results` memanfaatkan fungsionalitas pencarian gambar dan mengembalikan daftar jalur gambar yang mirip.

## Penggunaan
1. Siapkan dataset gambar, lalu simpan didalam folder static/img/dataset-image
2. Instal pustaka yang dibutuhkan dengan menggunakan `pip install -r requirements.txt` (asumsikan file `requirements.txt` disediakan).

3. Jalankan aplikasi Flask dengan menjalankan skrip utama: `python main.py`.

4. Akses aplikasi melalui browser web di `http://localhost:5000`.

5. Unggah gambar menggunakan antarmuka dan amati gambar yang mirip dalam hasil pencarian.

## Kesimpulan

Program pencarian kesamaan gambar menyediakan antarmuka yang intuitif bagi pengguna untuk menemukan gambar yang mirip dengan memanfaatkan kekuatan deep learning dan model Xception. Dokumentasi ini mencakup struktur, fungsionalitas, dan penggunaan program untuk membantu Anda memahami dan memanfaatkannya secara efektif.

## Docs are generate by Chat GPT

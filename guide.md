# Panduan Penggunaan Face Recognition Pipeline

Proyek ini memiliki **dua pendekatan** berbeda untuk mengenali wajah:
1. **Pendekatan Fitur Manual (Geometri & Tekstur)**: Mengekstrak rasio jarak mata, hidung, mulut, LBP, serta kerapatan tepi/kulit.
2. **Pendekatan Eigenface (PCA)**: Menggunakan analisis statistik (Principal Component Analysis) untuk memproyeksikan gambar wajah menjadi vektor *eigen*.

Kedua pendekatan ini merupakan murni *from-scratch* (menggunakan matriks matematika manual via `numpy`, tanpa bergantung pada algoritma *black-box* AI/ML bawaan seperti `cv2.face`).

---

## 1. Persiapan Data (Wajib untuk Kedua Pendekatan)

Sistem telah dirancang agar **Dataset Pelatihan (Training)** dan **Dataset Pengujian (Testing)** terpisah secara ketat untuk mencegah kebocoran data (*data leakage*).

1. Buka folder `webcam/`.
2. Masukkan foto mentah (raw/belum di-crop) yang akan dijadikan **data pelatihan**. Beri nama sesuai subjek (misal: `aldo (1).png`, `daffa (1).png`).
3. Masukkan foto mentah (raw/belum di-crop) yang akan dijadikan **data pengujian (testing)**. Pastikan **menambahkan kata "datatest"** pada namanya (misal: `aldo_gaya2_datatest.png`).
4. Jalankan script deteksi wajah:
   ```bash
   python deteksi_wajah.py
   ```
   *Script ini akan memotong wajah (cropping) menggunakan Haar Cascade.*
   * Data Pelatihan akan disimpan otomatis ke folder `hasil_deteksi/`.
   * Data Pengujian akan disimpan otomatis ke folder `datatest/`.

---

## PENDEKATAN 1: Pengenalan Berbasis Fitur Manual

Pendekatan ini akan menghitung jarak titik-titik biologis wajah (jarak mata ke hidung, proporsi mulut, dll) dan menyimpannya ke dalam file `.json`.

### Langkah 1: Ekstraksi Fitur Latih
Jalankan script berikut untuk mengekstrak fitur dari folder `hasil_deteksi/`:
```bash
python ekstraksi_fitur.py
```
> **Output:** Menghasilkan file `database_fitur.json`.

### Langkah 2: Ekstraksi Fitur Uji
Jalankan script berikut untuk mengekstrak fitur dari folder `datatest/`:
```bash
python ekstraksi_fitur_uji.py
```
> **Output:** Menghasilkan file `database_uji.json`.

### Langkah 3: Visualisasi Fitur (Opsional)
Jika Anda ingin melihat tingkat presisi dan seberapa akurat sistem menaruh titik di area hidung/mulut/mata, jalankan:
```bash
python simpan_visualisasi.py
```
> **Output:** Menggambar titik, kotak *bounding box*, serta garis kalkulasi geometri langsung di atas foto dan menyimpannya ke folder `hasil_visualisasi/`. (Alternatif: Gunakan `python visualisasi_overlay.py` jika ingin melihatnya lewat aplikasi *pop-up*).

### Langkah 4: Uji Pencocokan (Recognition)
Jalankan script pengujian untuk membandingkan kedua database menggunakan metode jarak Euclidean / K-Nearest Neighbor:
```bash
python demo_pengenalan_manual.py
```
> **Output:** Terminal akan menampilkan kecocokan wajah, status pengenalan (Benar/Salah), serta akurasi persentase akhir.

---

## PENDEKATAN 2: Pengenalan Berbasis Eigenface (PCA)

Pendekatan ini berfokus pada piksel gambar utuh (Holistic) dan **tidak menggunakan JSON** maupun fitur geometris. Sangat disarankan gambar crop memiliki posisi wajah/mata yang selaras (aligned) untuk hasil Eigenface yang maksimal.

*(Pastikan Anda sudah menjalankan `deteksi_wajah.py` pada Tahap 1 Persiapan Data di atas)*

### Langkah Tunggal: Training & Testing Sekaligus
Jalankan script berikut:
```bash
python eigenface.py
```
**Apa yang terjadi saat script ini dijalankan?**
1. **Training:** Membaca seluruh gambar crop dari folder `hasil_deteksi/`, meratakannya menjadi vektor 1D, lalu menghitung matriks rata-rata (*Mean Face*), matriks kovariansi, *Eigenvectors*, dan beban (weights).
2. **Testing:** Membaca gambar crop dari folder `datatest/`, memproyeksikannya ke ruang wajah (*Face Space*), dan menghitung jarak L2 (*Euclidean*) dengan gambar latih.
3. **Output:** Menampilkan prediksi beserta skor jarak langsung di terminal.

---

## Tips & Troubleshooting
* **Wajah Tidak Terdeteksi?** Jika Haar Cascade gagal menemukan wajah (karena cahaya buram/menunduk), `deteksi_wajah.py` tidak akan memotong foto dan menggunakan gambar utuh (*fallback full frame*).
* **Jarak Threshold:** Jika sistem terlalu sering memprediksi "TIDAK DIKENALI" padahal itu adalah orang yang sama, Anda bisa sedikit menaikkan nilai `threshold` pada script pencocokan.

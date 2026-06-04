# Histogram Specification Web App

Aplikasi web berbasis Flask yang mengimplementasikan algoritma Histogram Specification untuk pengolahan citra digital. Pengguna dapat mengupload dua gambar, yaitu gambar sumber dan gambar target, kemudian aplikasi akan memproses dan menyesuaikan distribusi intensitas piksel gambar sumber agar mengikuti distribusi histogram gambar target. Hasil yang ditampilkan mencakup gambar hasil transformasi, grafik histogram ketiga gambar, grafik CDF beserta perbandingannya, tabel inverse mapping, tabel verifikasi, nilai MSE, dan log proses algoritma secara lengkap.

---

## Persyaratan Sistem

- Python 3.9 atau lebih baru
- pip (sudah termasuk dalam instalasi Python standar)
- Koneksi internet untuk mengunduh dependencies

---

## Instalasi Python

Jika Python belum terinstal, unduh installer dari situs resmi:

```
https://www.python.org/downloads/
```

Pilih versi terbaru untuk sistem operasi yang digunakan. Saat proses instalasi di Windows, pastikan mencentang opsi "Add Python to PATH" sebelum klik Install.

Verifikasi instalasi Python berhasil dengan membuka terminal atau command prompt dan jalankan:

```bash
python --version
```

Verifikasi pip tersedia:

```bash
pip --version
```

---

## Cara Menjalankan Aplikasi

### 1. Clone atau download project

Jika menggunakan Git:

```bash
git clone <url-repository>
cd <nama-folder-project>
```

Jika download manual, ekstrak file ZIP lalu buka terminal di dalam folder project tersebut.

### 2. Buat virtual environment (disarankan)

Virtual environment memisahkan dependencies project ini dari instalasi Python global di komputer.

```bash
python -m venv venv
```

Aktifkan virtual environment:

Untuk Windows:
```bash
venv\Scripts\activate
```

Untuk macOS atau Linux:
```bash
source venv/bin/activate
```

Setelah aktif, prompt terminal akan menampilkan nama environment di depannya, misalnya `(venv)`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Proses ini akan mengunduh dan menginstal semua library yang dibutuhkan, yaitu Flask, OpenCV, NumPy, Matplotlib, dan Werkzeug. Tunggu hingga semua selesai terinstal.

Verifikasi instalasi berhasil:

```bash
pip list
```

Pastikan Flask, opencv-python-headless, numpy, dan matplotlib muncul dalam daftar.

### 4. Jalankan aplikasi

```bash
python app.py
```

Terminal akan menampilkan output seperti berikut:

```
* Serving Flask app 'app'
* Debug mode: on
* Running on http://127.0.0.1:5000
* Running on http://192.168.x.x:5000
```

### 5. Buka di browser

Akses aplikasi melalui browser dengan membuka alamat:

```
http://localhost:5000
```

Aplikasi siap digunakan. Untuk menghentikan server, tekan `Ctrl + C` di terminal.

---

## Cara Menggunakan Aplikasi

1. Pada halaman utama, klik tombol "Pilih Gambar" di kotak Gambar Asli (Source) dan pilih gambar yang ingin diproses.
2. Klik tombol "Pilih Gambar" di kotak Gambar Target dan pilih gambar referensi yang histogramnya ingin ditiru.
3. Setelah kedua gambar dipilih, tombol "Proses Histogram Specification" akan aktif. Klik tombol tersebut.
4. Tunggu proses selesai. Untuk gambar berukuran besar, proses mungkin membutuhkan beberapa detik.
5. Hasil akan ditampilkan secara otomatis di bawah, mencakup gambar hasil, grafik histogram, grafik CDF, tabel mapping, tabel verifikasi, nilai MSE, dan log proses.

---

## Struktur Project

```
project/
├── api/
│   └── index.py          # Entry point untuk deployment Vercel
├── templates/
│   └── index.html        # Halaman antarmuka pengguna
├── app.py                # Backend Flask dan seluruh logika algoritma
├── requirements.txt      # Daftar dependencies Python
├── vercel.json           # Konfigurasi deployment Vercel
└── .gitignore
```

---


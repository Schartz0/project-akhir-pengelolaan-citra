# Panduan Penggunaan Aplikasi Coral Health Analyzer

## Menjalankan Aplikasi

### 1. Persiapan
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
python app.py
```

### 2. Akses Aplikasi
Buka browser dan akses: **http://localhost:5000**

## Cara Menggunakan

### Langkah 1: Upload Gambar
1. Klik area upload atau seret gambar terumbu karang ke dalam kotak upload
2. Format yang didukung: JPG, PNG, WEBP (maks 20MB)
3. Preview gambar akan muncul di bawah area upload beserta nama file dan ukurannya

### Langkah 2: Analisis
1. Klik tombol **"Analisis Sekarang"**
2. Tunggu proses selesai — loading spinner akan muncul selama model berjalan
3. Hasil akan ditampilkan otomatis dan halaman scroll ke bagian hasil

### Langkah 3: Baca Hasil
Lihat penjelasan setiap bagian hasil di bawah.

### Langkah 4: Reset
Klik **"← Analisis Gambar Lain"** untuk menganalisis gambar baru.

---

## Memahami Hasil

### 1. Verdict Card (Hasil Prediksi)
Kotak berwarna besar yang menampilkan:
- **Status terumbu karang** dalam Bahasa Indonesia (Sehat / Tidak Sehat / Mati)
- **Nama kelas Inggris** di bawahnya (Healthy / Unhealthy / Dead)
- **Confidence** — seberapa yakin model terhadap prediksi ini (dalam %)
- **Timestamp** — waktu prediksi dilakukan

Warna kartu:
- 🟢 Hijau = Sehat (Healthy)
- 🟡 Kuning = Tidak Sehat (Unhealthy)
- 🔴 Merah = Mati (Dead)

### 2. Distribusi Probabilitas
Dua tampilan untuk distribusi probabilitas ketiga kelas:

**Progress Bars:**
- Bar horizontal per kelas dengan persentase di kanan
- Kelas prediksi ditampilkan dengan teks tebal

**Donut Chart (Gauge):**
- Lingkaran dengan persentase confidence di tengah
- Warna sesuai kelas prediksi

**Bar Chart:**
- Grafik batang perbandingan ketiga kelas
- Bar kelas prediksi berwarna, sisanya abu-abu

### 3. Gambar Original vs Setelah Preprocessing
Dua gambar side-by-side:
- **Gambar Original**: Gambar yang diunggah apa adanya
- **Setelah Preprocessing**: Gambar setelah white balance, dehazing, dan CLAHE

Perhatikan perbedaan warna dan kontras — preprocessing dirancang untuk mengoreksi kondisi bawah air.

### 4. Analisis Histogram Preprocessing
Grafik 4 panel:
- **Baris atas**: Tampilan gambar (original | preprocessed)
- **Baris bawah**: Histogram RGB (original | preprocessed)

Histogram memperlihatkan distribusi intensitas channel R, G, B sebelum dan sesudah preprocessing.

### 5. Pipeline Preprocessing
Daftar 4 langkah teknis yang dijalankan sebelum inferensi:
1. White Balance (koreksi warna LAB)
2. Dehazing (Dark Channel Prior)
3. CLAHE Enhancement
4. Resize & Normalize (224×224, ImageNet stats)

---

## Interpretasi Kelas

| Kelas | Kondisi | Indikator Visual |
|-------|---------|-----------------|
| **Sehat** | Terumbu karang hidup, warna cerah, struktur utuh | Warna coral/orange/ungu cerah |
| **Tidak Sehat** | Pemutihan parsial atau tekanan lingkungan | Warna pucat sebagian, struktur masih ada |
| **Mati** | Terumbu karang sudah tidak hidup | Dominan putih/abu, tidak ada warna hidup |

---

## Tips Penggunaan

### Kualitas Gambar
- Gunakan gambar dengan pencahayaan cukup
- Hindari gambar yang terlalu buram atau gelap
- Resolusi minimum yang disarankan: 224×224 piksel
- Model dilatih dengan gambar underwater — hasil terbaik pada foto bawah air

### Confidence Rendah
Jika confidence < 60%, pertimbangkan:
- Gambar terlalu buram atau terdistorsi
- Objek dalam gambar bukan terumbu karang
- Sudut pengambilan tidak ideal

### Batch Analisis
Untuk menganalisis banyak gambar, ulangi proses upload satu per satu menggunakan tombol reset di bawah hasil.

---

## Troubleshooting

### Error: "Gambar harus diupload"
- Pastikan sudah memilih atau menyeret file gambar ke area upload

### Error: "Gagal membaca gambar"
- Pastikan file adalah gambar valid (JPG/PNG/WEBP)
- Coba dengan file lain untuk memastikan format didukung

### Proses Lambat
- Model pertama kali di-load saat startup, request berikutnya lebih cepat
- Gunakan CPU yang lebih cepat atau aktifkan GPU (CUDA) untuk inferensi lebih cepat

### Model Tidak Ditemukan
- Pastikan file `best_coral_model.pth` ada di direktori yang sama dengan `app.py`

---

## Referensi Teknis

- **Arsitektur Model**: ResNet-50 (He et al., 2016)
- **Dehazing**: Dark Channel Prior (He et al., 2011)
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization (Zuiderveld, 1994)
- **Framework**: PyTorch, albumentations, OpenCV

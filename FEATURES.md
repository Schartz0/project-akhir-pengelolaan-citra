# Fitur Lengkap Aplikasi Coral Health Analyzer

## Fitur Utama

### 1. Upload Gambar
- Upload gambar terumbu karang (drag & drop atau klik)
- Preview gambar sebelum dianalisis
- Support format: JPG, PNG, WEBP
- Maksimal ukuran file: 20MB

### 2. Pipeline Preprocessing Underwater

Sebelum gambar masuk ke model, dilakukan 4 tahap preprocessing otomatis:

#### Tahap 1: White Balance
- Koreksi warna menggunakan LAB color space
- Menghilangkan color cast biru/hijau khas fotografi bawah air

#### Tahap 2: Dehazing (Dark Channel Prior)
- Estimasi atmospheric light dari dark channel image
- Menghitung transmission map untuk koreksi kekeruhan air
- Parameter: omega=0.95, t0=0.1
- Rumus: J(x) = (I(x) - A) / t(x) + A

#### Tahap 3: CLAHE Enhancement
- Contrast Limited Adaptive Histogram Equalization
- clipLimit=2.0, tileGridSize=8×8
- Meningkatkan detail dan kontras lokal tanpa over-amplification

#### Tahap 4: Resize & Normalize
- Resize ke 224×224 piksel (input ResNet-50)
- Normalisasi dengan mean=[0.485, 0.456, 0.406] dan std=[0.229, 0.224, 0.225] (ImageNet)

### 3. Inferensi Model Deep Learning

- **Arsitektur**: ResNet-50 (fine-tuned)
- **Output**: 3 kelas kesehatan terumbu karang
- **Framework**: PyTorch
- **Device**: GPU (CUDA) jika tersedia, fallback ke CPU

#### Kelas Prediksi:
| Kelas | Label Indonesia | Keterangan |
|-------|----------------|------------|
| Healthy | Sehat | Terumbu karang dalam kondisi baik |
| Unhealthy | Tidak Sehat | Terumbu karang mengalami tekanan/pemutihan parsial |
| Dead | Mati | Terumbu karang sudah tidak hidup |

### 4. Visualisasi Hasil

#### A. Verdict Card
- Label kelas dalam Bahasa Indonesia
- Warna indikator status (hijau / kuning / merah)
- Nilai confidence dalam persen
- Timestamp prediksi

#### B. Distribusi Probabilitas
- Progress bar per kelas (Sehat / Tidak Sehat / Mati)
- Persentase tiap kelas ditampilkan secara real-time
- Donut chart confidence (gauge)
- Bar chart perbandingan ketiga kelas (matplotlib)

#### C. Perbandingan Gambar
- Gambar original vs gambar setelah preprocessing (side-by-side)

#### D. Analisis Histogram Preprocessing
- Histogram RGB gambar original vs setelah preprocessing
- Visualisasi 4 panel: gambar + histogram per sisi
- Memperlihatkan efek koreksi warna dan peningkatan kontras

#### E. Pipeline Steps
- Penjelasan 4 langkah preprocessing yang dijalankan
- Deskripsi teknis setiap tahap

## Desain UI

- Style minimal hitam-putih, konsisten dan bersih
- Responsive design (mobile-friendly)
- Drag & drop upload dengan visual feedback
- Loading indicator dengan spinner
- Error handling dengan pesan jelas
- Tombol reset untuk analisis gambar baru

## Performa

- Model di-load sekali saat startup (tidak reload per request)
- Preprocessing menggunakan OpenCV (cepat)
- Inferensi dengan `torch.no_grad()` untuk efisiensi memori
- Gambar di-resize sebelum encoding base64 untuk transfer ringan

## Output per Analisis

Setiap analisis menghasilkan:
1. Verdict kelas + confidence
2. 3 probability bars
3. 1 donut chart confidence
4. 1 bar chart probabilitas
5. 2 gambar (original + preprocessed)
6. 1 grafik histogram perbandingan (4 panel)
7. Ringkasan pipeline preprocessing

## Teknologi

- **Backend**: Flask (Python)
- **Model**: PyTorch + ResNet-50
- **Preprocessing**: OpenCV, albumentations
- **Visualisasi**: Matplotlib
- **Numerical Computing**: NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## Kompatibilitas

- Desktop: Chrome, Firefox, Safari, Edge
- Mobile: iOS Safari, Chrome Mobile
- Tablet: iPad, Android Tablet

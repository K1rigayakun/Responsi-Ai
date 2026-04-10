# 🐾 Animal Classifier — CNN AFHQ

Website klasifikasi hewan **(Kucing / Anjing / Hewan Liar)** berbasis model CNN PyTorch yang berjalan **langsung di browser** via ONNX Runtime Web — tanpa server, tanpa API, 100% client-side.

> Dibuat oleh **Michael Deryl Aaron Matthew** (NIM: 241712042, Kelas A2) sebagai tugas Responsi AI.

**[🌐 Buka Website Demo →](https://k1rigayakun.github.io/Responsi-Ai/)**

---

## 📁 Struktur Repository

```
Animal-Classifier/
├── index.html      ← Website classifier (pure HTML/CSS/JS)
├── model.onnx      ← Model CNN ter-export (upload manual setelah training)
└── README.md
```

---

## 🧠 Optimalisasi Model (Anti-Overfitting)

Model original (`animal_classifier.ipynb`) memiliki beberapa masalah yang menyebabkan **overfitting** — model hafal data training tapi gagal generalisasi ke data baru. Berikut 9 perbaikan yang dilakukan:

---

### 1. Reproducibility — Penambahan `random_state`

| | Sebelum | Sesudah |
|---|---|---|
| Split data | `data_df.sample(frac=0.7)` — acak setiap run | `data_df.sample(frac=0.7, random_state=42)` — selalu sama |
| Seed global | Tidak ada | `torch.manual_seed(42)` + `np.random.seed(42)` |

Tanpa `random_state`, setiap kali kode dijalankan ulang pembagian data train/val/test berbeda sehingga hasil eksperimen tidak bisa dibandingkan secara konsisten.

---

### 2. Augmentasi Data Training *(Perbaikan Overfitting Utama)*

| | Sebelum | Sesudah |
|---|---|---|
| Transform | Satu transform sama untuk semua | Dua transform terpisah: train vs val/test |
| Augmentasi | Tidak ada | `RandomHorizontalFlip` + `RandomRotation(15°)` + `ColorJitter` |
| Normalisasi | `ConvertImageDtype` saja | `Normalize` dengan mean/std ImageNet standar |

```python
# Training — dengan augmentasi agar model tidak hafal piksel
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Val/Test — tanpa augmentasi agar evaluasi konsisten
val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Augmentasi adalah teknik paling efektif mengatasi overfitting tanpa menambah data baru — model dipaksa belajar fitur yang robust, bukan menghafal piksel.

---

### 3. Batch Size, Epochs, dan Shuffle Val/Test

| | Sebelum | Sesudah |
|---|---|---|
| `BATCH_SIZE` | 16 | **32** — gradien lebih stabil |
| `EPOCHS` | 10 (fixed) | **30** — dikontrol Early Stopping |
| `val_loader` shuffle | `True` | **`False`** — evaluasi konsisten |
| `test_loader` shuffle | `True` | **`False`** — evaluasi konsisten |

Batch size lebih besar menghasilkan estimasi gradien yang lebih akurat. Epochs yang dikontrol Early Stopping mencegah model train terlalu lama dan overfitting.

---

### 4. Arsitektur Model — BatchNorm + Dropout + Conv Block Tambahan

| | Sebelum | Sesudah |
|---|---|---|
| Conv blocks | 3 block | **4 block** |
| BatchNorm | Tidak ada | **`BatchNorm2d`** setelah setiap Conv layer |
| Dropout | Tidak ada | **`Dropout(p=0.5)`** di fully connected layer |
| FC layers | `Linear(128×16×16, 128)` | `Linear(256×8×8, 512)` → `Linear(512, 128)` |

```
Input [1, 3, 128, 128]
  ↓ Conv2d(3→32)   + BN + ReLU + MaxPool  →  [1,  32, 64, 64]
  ↓ Conv2d(32→64)  + BN + ReLU + MaxPool  →  [1,  64, 32, 32]
  ↓ Conv2d(64→128) + BN + ReLU + MaxPool  →  [1, 128, 16, 16]
  ↓ Conv2d(128→256)+ BN + ReLU + MaxPool  →  [1, 256,  8,  8]
  ↓ Flatten  →  [1, 16384]
  ↓ Linear(16384→512) + BN + ReLU + Dropout(0.5)
  ↓ Linear(512→128) + ReLU
  ↓ Linear(128→3)
Output [1, 3]  ←  raw logits
```

**BatchNorm** menstabilkan distribusi aktivasi antar layer sehingga training lebih cepat dan stabil. **Dropout(0.5)** mematikan 50% neuron secara acak setiap forward pass, mencegah neuron saling bergantung (*co-adaptation*) yang merupakan penyebab utama overfitting pada FC layer.

---

### 5. L2 Regularization + Learning Rate Scheduler

| | Sebelum | Sesudah |
|---|---|---|
| Optimizer | `Adam(lr=1e-4)` | `Adam(lr=1e-4, weight_decay=1e-4)` |
| LR Scheduler | Tidak ada | `ReduceLROnPlateau(factor=0.5, patience=3)` |

```python
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
```

`weight_decay=1e-4` menambahkan penalti L2 pada bobot besar, mendorong model menggunakan bobot yang lebih kecil dan general. `ReduceLROnPlateau` otomatis memangkas LR sebesar 50% jika val_loss tidak membaik selama 3 epoch berturut-turut — membantu model keluar dari plateau tanpa overshoot.

---

### 6. Early Stopping

| | Sebelum | Sesudah |
|---|---|---|
| Mekanisme stop | Training jalan penuh sampai epoch habis | Berhenti otomatis jika val_loss stagnan 5 epoch |
| Simpan model | Tidak ada checkpoint | Simpan otomatis model terbaik ke `best_model.pt` |

```python
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001, path='best_model.pt'):
        ...
    def __call__(self, val_loss, model):
        if improved: self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

# Di akhir training loop:
model.load_state_dict(torch.load('best_model.pt'))  # load model terbaik
```

Early Stopping memastikan model berhenti di titik generalisasi terbaik, bukan di epoch akhir yang kemungkinan sudah overfitting.

---

### 7. `model.train()` dan `model.eval()`

| | Sebelum | Sesudah |
|---|---|---|
| Training phase | Tidak ada `model.train()` | `model.train()` — Dropout aktif, BN update stats |
| Validation phase | Tidak ada `model.eval()` | `model.eval()` — Dropout nonaktif, BN pakai running stats |
| Fungsi prediksi | Tidak ada `model.eval()` | `model.eval()` wajib sebelum prediksi |

Ini **bukan opsional** — tanpa `model.eval()` saat validasi, Dropout masih mematikan neuron secara acak sehingga metrik val/test tidak deterministik dan tidak mencerminkan performa nyata model.

---

### 8. Normalisasi Loss (Dibagi `len(loader)`, bukan 1000)

| | Sebelum | Sesudah |
|---|---|---|
| Normalisasi | Hardcoded `/ 1000` untuk semua kondisi | `/ len(train_loader)` — jumlah batch aktual |

```python
# Sebelum (salah):
total_loss_train_plot.append(round(total_loss_train / 1000, 4))

# Sesudah (benar):
avg_train_loss = total_loss_train / len(train_loader)
total_loss_train_plot.append(round(avg_train_loss, 4))
```

Nilai 1000 adalah angka hardcoded yang tidak berdasarkan jumlah batch aktual. Jika batch size atau jumlah data berubah, grafik loss menjadi tidak representatif dan tidak bisa dibandingkan antar eksperimen.

---

### 9. Perbaikan Bug `predict_image()`

| | Sebelum | Sesudah |
|---|---|---|
| Bug utama | Parameter `image_path` **tidak digunakan** | Parameter `image_path` benar-benar digunakan |
| Path gambar | Hardcoded ke `/content/cat.jpg` | Dibaca dari parameter input |
| `model.eval()` | Tidak dipanggil — Dropout masih aktif | Dipanggil sebelum prediksi |
| Transform | Pakai `transform` (sama dengan training) | Pakai `val_test_transform` (tanpa augmentasi) |
| Output | Hanya label prediksi | Label + confidence + bar probabilitas semua kelas |

Bug ini menyebabkan `predict_image()` selalu memprediksi gambar `cat.jpg` yang sama, apapun file yang diinput.

---

## 🌐 Cara Kerja Website

Website menjalankan model CNN **langsung di browser** via ONNX Runtime Web — tanpa server backend, tanpa API call, semua inferensi terjadi di sisi klien menggunakan WebAssembly (WASM).

### Alur Inferensi

```
Halaman Dibuka
      ↓
loadModel() → fetch("./model.onnx") → ort.InferenceSession.create()
      ↓
User Upload Gambar (JPG/PNG/WEBP)
      ↓
FileReader API → preview gambar (tanpa kirim ke server)
      ↓
preprocessImage():
  Canvas HTML5 → resize 128×128
  RGBA uint8 → float32 CHW tensor
  Normalize: (pixel/255 - mean) / std  ← sama dengan val_test_transform
      ↓
ort.Tensor("float32", floatData, [1, 3, 128, 128])
      ↓
session.run() → Forward pass CNN via WASM
      ↓
Raw Logits [3]  →  Softmax  →  Probabilitas [cat, dog, wild]
      ↓
Tampilkan: verdict box + animated probability bars + raw logits
```

### Step-by-Step

**Step 1 — Load Model ONNX**
Saat halaman dibuka, `loadModel()` otomatis mengambil `model.onnx` dari server via `fetch()`. Ada 3 strategi fallback jika loading gagal (WASM SIMD on → SIMD off → versi lama) untuk kompatibilitas maksimal di berbagai browser.

**Step 2 — Upload Gambar**
User upload lewat klik atau drag-and-drop. `FileReader API` membaca file secara lokal tanpa mengirimkannya ke server manapun.

**Step 3 — Preprocessing**
Gambar di-resize ke 128×128 menggunakan Canvas HTML5, lalu dikonversi dari format RGBA uint8 ke float tensor CHW dan dinormalisasi dengan mean/std ImageNet — **identik** dengan `val_test_transform` di notebook Python.

**Step 4 — Inferensi**
Tensor dimasukkan ke ONNX Runtime Web. Model melakukan forward pass melalui 4 conv block + FC layers dan menghasilkan 3 nilai logit (satu per kelas).

**Step 5 — Softmax & Tampilkan Hasil**
Logit dikonversi ke probabilitas via softmax. Kelas dengan probabilitas tertinggi ditampilkan sebagai prediksi beserta animated probability bars dan raw logit values.

---

## 🚀 Cara Deploy

### Step 1 — Export model ke ONNX (di Google Colab)

Jalankan cell ini setelah training selesai:

```python
import torch, os

model.eval()
dummy_input = torch.randn(1, 3, 128, 128).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,        # opset 11 paling stabil
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamo=False             # paksa legacy exporter agar bobot tersimpan
)

size_mb = os.path.getsize("model.onnx") / 1024 / 1024
print(f"✅ Ukuran: {size_mb:.1f} MB")   # harusnya > 30 MB
print(f"Kelas: {list(label_encoder.classes_)}")  # ['cat', 'dog', 'wild']

from google.colab import files
files.download("model.onnx")
```

> **⚠️ Penting:** Jika ukuran file < 1 MB, bobot tidak tersimpan. Pastikan `dynamo=False` dan `export_params=True`.

### Step 2 — Upload ke Repository

1. Buka repository di GitHub
2. Upload file `model.onnx` yang baru di-download ke root folder

### Step 3 — Aktifkan GitHub Pages

1. Buka **Settings → Pages**
2. Source: `Deploy from a branch`
3. Branch: `main`, folder: `/ (root)`
4. Klik **Save** — tunggu ~1 menit
5. Website live di `https://USERNAME.github.io/REPO-NAME`

---

## 📦 Tech Stack

| Komponen | Detail |
|---|---|
| Model | CNN PyTorch — 8.8M parameters |
| Export | ONNX (opset 11, legacy exporter) |
| Runtime | [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) v1.17.3 via CDN |
| Execution | WebAssembly (WASM) — 100% client-side |
| Frontend | Pure HTML/CSS/JS — zero dependencies |
| Dataset | [AFHQ Animal Faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces) (Kaggle) |

---

## 📊 Kelas Output

| Index | Label | Keterangan |
|---|---|---|
| 0 | `cat` | 🐱 Kucing |
| 1 | `dog` | 🐶 Anjing |
| 2 | `wild` | 🦁 Hewan Liar |

Urutan berdasarkan `LabelEncoder` sklearn yang sort alfabetis.

---

*Repository: [github.com/K1rigayakun/Responsi-Ai](https://github.com/K1rigayakun/Responsi-Ai)*

# 🐾 Animal Classifier — CNN AFHQ

Website klasifikasi hewan (Kucing / Anjing / Hewan Liar) menggunakan model CNN PyTorch yang berjalan **langsung di browser** via ONNX Runtime Web.

> **Tidak ada server. Tidak ada API. 100% offline setelah model di-load.**

## 🌐 Demo
[Buka Website →](https://k1rigayakun.github.io/Responsi-Ai/)

---

## 📁 Struktur Repository

```
Animal-Classifier/
├── index.html          ← Website utama
├── model.onnx          ← Model CNN yang sudah di-export (kamu upload manual)
├── export_onnx.py      ← Script cell untuk export model dari notebook
└── README.md
```

---

## 🚀 Cara Deploy

### Step 1 — Export model ke ONNX (di Google Colab)

Tambahkan cell berikut di **akhir notebook** (`Animal Classifier Optimized.ipynb`), jalankan setelah training selesai:

```python
import torch, os

model.eval()
dummy_input = torch.randn(1, 3, 128, 128).to(device)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ Ukuran: {os.path.getsize('model.onnx')/1024/1024:.1f} MB")
print(f"Kelas: {list(label_encoder.classes_)}")  # harus ['cat', 'dog', 'wild']

from google.colab import files
files.download("model.onnx")
```

### Step 2 — Upload ke Repository

1. Clone atau buka repository ini di GitHub
2. Upload file `model.onnx` yang baru di-download
3. Pastikan struktur file seperti di atas

### Step 3 — Aktifkan GitHub Pages

1. Buka **Settings** → **Pages**
2. Source: `Deploy from a branch`
3. Branch: `main`, folder: `/ (root)`
4. Klik **Save**
5. Tunggu ~1 menit → website live di `https://USERNAME.github.io/REPO-NAME`

---

## ⚙️ Cara Kerja

```
Gambar JPG/PNG/WEBP
        ↓
   Resize 128×128 px          ← sama dengan val_test_transform di notebook
        ↓
 Normalize (ImageNet           mean=[0.485,0.456,0.406]
 mean & std)                   std=[0.229,0.224,0.225]
        ↓
 Float32 Tensor [1,3,128,128]  ← format CHW seperti PyTorch
        ↓
   ONNX Runtime Web            ← model.onnx berjalan di WASM
        ↓
  Raw Logits [3]               ← output layer Linear(128, 3)
        ↓
     Softmax                   ← probabilitas cat / dog / wild
        ↓
    Hasil Prediksi
```

**Kelas output (urutan alfabetis dari LabelEncoder):**
- Index 0 → `cat` (Kucing)
- Index 1 → `dog` (Anjing)
- Index 2 → `wild` (Hewan Liar)

---

## 🏗️ Arsitektur Model

```
Input [1, 3, 128, 128]
  ↓ Conv2d(3→32) + BN + ReLU + MaxPool → [1, 32, 64, 64]
  ↓ Conv2d(32→64) + BN + ReLU + MaxPool → [1, 64, 32, 32]
  ↓ Conv2d(64→128) + BN + ReLU + MaxPool → [1, 128, 16, 16]
  ↓ Conv2d(128→256) + BN + ReLU + MaxPool → [1, 256, 8, 8]
  ↓ Flatten → [1, 16384]
  ↓ Linear(16384→512) + BN + ReLU + Dropout(0.5)
  ↓ Linear(512→128) + ReLU
  ↓ Linear(128→3)
Output [1, 3]  ← raw logits
```

---

## 📦 Dependencies

- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) v1.18 (via CDN)
- Tidak ada dependency lain — pure HTML/CSS/JS

---

Dataset: [AFHQ Animal Faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

# 🌊 Underwater Image Enhancement via Knowledge Distillation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20T4%20x2-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Sea--thru-0077B6?style=for-the-badge)

**Restorasi Warna dan Kontras pada Citra Bawah Air dengan Knowledge Distillation**

*Deep Learning · RGB-D · Underwater Vision · AUV Deployment*

</div>

---

## 🐠 Overview

Citra bawah air sering mengalami degradasi warna, kontras rendah, dan haze akibat penyerapan cahaya.  
Proyek ini menggunakan strategi **Knowledge Distillation (KD)**, yaitu model Teacher mentransfer pengetahuan ke model Student yang lebih ringan sehingga cocok untuk deployment real-time pada perangkat seperti **Autonomous Underwater Vehicle (AUV)**.

```text
Input (Buram/Hijau) → Student Model → Output (Jernih/Natural)
```

---

## 👥 Tim — Kelompok 3

| No | Nama | NIM |
|:--:|------|:---:|
| 1 | Cheryl Eugene Saari | 2802434844 |
| 2 | Dasnaiya Hsu | 2802420901 |
| 3 | Ernestine Danella Ong | 2802421135 |
| 4 | Stephanie Renata Panghudi | 2802416822 |
| 5 | Annabelle Frederica Suryana | 2802412351 |

---

## 🗂️ Pembagian Tugas

| Kombinasi | Nama | Teacher | Student | Fokus |
|-----------|------|---------|---------|-------|
| **Preprocessing** | Annabelle Frederica Suryana | - | - | Data loading, depth refinement, patch extraction, augmentation |
| **Combo 1** | Cheryl Eugene Saari | UGAN | GhostNet-UNet | Kualitas Visual |
| **Combo 2** | Ernestine Danella Ong | FUnIE-GAN | MobileNetV3 | Kecepatan Ekstrem |
| **⭐ Combo 3 (Main Model)** | Stephanie Renata Panghudi | Water-Net | Shallow U-Net | Balanced Enhancement |
| **Combo 4** | Dasnaiya Hsu | U-Transformer | MobileViT | Global Color Cast |
| **Combo 5** | Dasnaiya Hsu | DeepSESR | Tiny-GAN | Ketajaman Objek |

---

## 📊 Dataset

**Sea-thru Dataset**  
https://www.kaggle.com/datasets/colorlabeilat/seathru-dataset

Dataset RGB-D (RGB image + depth map) yang diambil di lingkungan laut nyata dan terdiri dari beberapa scene bawah air.

| Split | Persentase |
|-------|------------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

## 🔧 Preprocessing Pipeline

```text
RGB Image + Depth Map
        ↓
Build RGB-Depth Pairs
        ↓
Global Depth Normalization
        ↓
RGB + Depth Loader (OpenCV)
        ↓
Patch Extraction (256×256)
        ↓
Data Augmentation
        ↓
tf.data Pipeline
```

### Tahapan Utama
- Pairing RGB dan depth map dilakukan sebelum training
- RGB dinormalisasi ke range `[-1,1]`
- Depth map diperbaiki menggunakan `cv2.inpaint()`
- Patch extraction menggunakan ukuran `256×256`
- Augmentasi dilakukan sinkron pada RGB, depth, dan valid mask
- Dataset diproses menggunakan `tf.data.Dataset`

---

## 🤖 Model Architecture

## Combo 1 — Feature Mimic

```text
Teacher : UGAN
Student : GhostNet U-Net

Encoder → Ghost Modules → Decoder
           ↓
   Lightweight Feature Extraction
```

Fokus pada transfer feature representation dari model besar ke model ringan menggunakan Ghost Modules.

---

## Combo 2 — Efficiency Champion

```text
Teacher : FUnIE-GAN
Student : MobileNetV3-Small Generator

Depthwise Conv → Mobile Blocks → Lightweight Decoder
```

Difokuskan untuk inference cepat dan deployment pada hardware ringan.

---

## ⭐ Combo 3 — Main Model

```text
Teacher : Water-Net
Student : Shallow U-Net (3-Level)

Encoder → Bottleneck → Decoder
      ↘ Skip Connections ↗
```

Kombinasi ini dipilih sebagai model utama karena menghasilkan enhancement warna dan kontras yang paling konsisten dibanding model lainnya.

---

## Combo 4 — Attention Transfer

```text
Teacher: U-Transformer
  ┌─ CNN Encoder (e1→e4) ─── Transformer Bottleneck ─── CNN Decoder ─┐
  └─────────────── Skip Connections (U-Net style) ────────────────────┘
  Input: 4ch (RGB + Depth) → Output: 3ch Enhanced RGB

Student: MobileViT
  ┌─ DW-Sep Conv ─── MobileViT Block (local 8×8 attention) ─── Decoder ─┐
  └──────────────── ~70% lebih ringan dari Teacher ───────────────────────┘
```

Menggunakan attention mechanism untuk memperbaiki global color cast pada citra bawah air.

---

## Combo 5 — Sharpness Specialist

```text
Teacher : DeepSESR
Dense Blocks → Residual Blocks ×16 → Reconstruction Head

Student : Tiny-GAN
Encoder → Decoder → PatchGAN Discriminator
```

Difokuskan untuk meningkatkan ketajaman objek dan edge preservation.

---

## 📈 Hasil Evaluasi

| Model | PSNR ↑ | SSIM ↑ |
|------|:------:|:------:|
| Combo 4 — MobileViT | 40.94 | 0.9261 |
| Combo 5 — Tiny-GAN | **43.85** | **0.9640** |

### Observasi
- Kombinasi 3 menghasilkan enhancement visual paling konsisten
- Kombinasi 5 menghasilkan PSNR dan SSIM tertinggi
- Beberapa model ringan memiliki inference cepat tetapi enhancement masih terbatas

---

## 🚀 Cara Menjalankan

### Environment
- Python 3.10
- TensorFlow 2.x
- Kaggle GPU T4 ×2

### Setup
1. Add dataset `Sea-thru`
2. Aktifkan GPU
3. Upload notebook `.ipynb`
4. Jalankan notebook secara berurutan

```python
TRAIN_STEPS = 500
VAL_STEPS   = 100
EPOCHS      = 10
BATCH_SIZE  = 4
```

---

## 📁 Struktur File

```text
project/
├── full_notebook_combo4_5.ipynb
├── kombinasi5_only.ipynb
├── project_revised_fixed.ipynb
├── README.md
└── /kaggle/working/
    ├── best_student_combo4.weights.h5
    ├── best_generator_combo5.weights.h5
    ├── before_after_combo4.png
    ├── before_after_combo5.png
    └── comparison_chart.png
```

---

## ⚠️ Tantangan

- Missing values pada depth map
- GPU memory terbatas saat training
- Trade-off antara kualitas dan efisiensi
- Domain gap antar lingkungan bawah air

---

## 🔮 Future Work

- Multi-domain training
- Physics-based + Deep Learning enhancement
- Quantization dan ONNX export untuk AUV
- Pengembangan Transformer yang lebih ringan
- Integrasi dengan object detection dan tracking

---

<div align="center">

Made with 🌊 by **Kelompok 3 — Binus University**

*"From murky depths to crystal clarity"*

</div>

# 🌊 Underwater Image Enhancement via Knowledge Distillation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20T4%20x2-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Sea--thru-0077B6?style=for-the-badge)

**Restorasi Warna dan Kontras pada Citra Bawah Air Menggunakan Jaringan Residual Dalam**

*Deep Learning · Knowledge Distillation · Underwater Vision · AUV Deployment*

</div>

---

## 🐠 Overview

Lingkungan bawah air menghadirkan tantangan optik yang unik — cahaya merah menghilang pertama kali, diikuti kuning dan oranye, meninggalkan gambar yang didominasi warna biru-hijau dengan kabut partikel tersuspensi. Proyek ini membangun sistem peningkatan citra bawah air menggunakan strategi **Knowledge Distillation**, di mana model Teacher yang besar dan akurat "mengajarkan" model Student yang ringan — ideal untuk deployment real-time pada **Autonomous Underwater Vehicle (AUV)**.

```
Input (Buram/Hijau)  →  [Student Model]  →  Output (Jernih/Natural)
        🌊                    🤖                      🐙
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
| **Combo 1** | Cheryl | UGAN | GhostNet-UNet | Kualitas Visual |
| **Combo 2** | Ernestine | FUnIE-GAN | MobileNetV3 | Kecepatan Ekstrem |
| **Combo 3** | Stephanie | Water-Net | Shallow U-Net | Physics Learner |
| **Combo 4** | Dasnaiya | U-Transformer | MobileViT | Global Color Cast |
| **Combo 5** | Dasnaiya | DeepSESR | Tiny-GAN | Ketajaman Objek |

---

## 📊 Dataset

**Sea-thru Dataset** — [`colorlabeilat/seathru-dataset`](https://www.kaggle.com/datasets/colorlabeilat/seathru-dataset)

Dataset RGB-D (gambar + depth map) yang diambil di lingkungan laut nyata, terdiri dari 5 scene berbeda (D1–D5).

| Split | Gambar | Patches (256×256) |
|-------|--------|-------------------|
| Train | ~620   | 44,164 |
| Val   | ~109   | 7,779  |
| Test  | ~128   | 9,219  |

---

## 🔧 Preprocessing Pipeline

```
RAW linearPNG  →  Load (16-bit)  →  Normalize [-1, 1]
Depth TIFF     →  In-painting    →  Normalize [0, 1]
               →  Patch 256×256  →  Augmentasi (flip, rotate)
               →  tf.data.Dataset (with .repeat())
```

**Langkah kunci:**
- Normalisasi RGB ke `[-1, 1]` (optimal untuk Tanh activation)
- Depth map in-painting dengan `cv2.INPAINT_NS` untuk mengisi area kosong
- Global depth normalization menggunakan `GLOBAL_DEPTH_MAX` seluruh dataset
- Augmentasi **sinkron** antara RGB dan depth (tidak color jitter!)
- Skip patch dengan valid depth coverage < 5%

---

## 🤖 Model Architecture

### Kombinasi 4 — Attention Transfer

> *Mengatasi global color cast menggunakan Transformer Attention*

```
Teacher: U-Transformer
  ┌─ CNN Encoder (e1→e4) ─── Transformer Bottleneck ─── CNN Decoder ─┐
  └─────────────── Skip Connections (U-Net style) ────────────────────┘
  Input: 4ch (RGB + Depth) → Output: 3ch Enhanced RGB

Student: MobileViT
  ┌─ DW-Sep Conv ─── MobileViT Block (local 8×8 attention) ─── Decoder ─┐
  └──────────────── ~70% lebih ringan dari Teacher ───────────────────────┘
```

### Kombinasi 5 — Sharpness Specialist

> *Menjaga ketajaman tepi objek untuk inspeksi AUV*

```
Teacher: DeepSESR
  Dense Blocks → Residual Blocks (×16) → Reconstruction Head

Student: Tiny-GAN
  Generator  : Encoder-Decoder + Skip Connections + Sobel Edge Loss
  Discriminator: PatchGAN (70×70 receptive field)
```

**Knowledge Distillation Loss:**

```
L_total = λ_rec · L1(pred, GT) + λ_dist · MSE(pred, teacher) + λ_per · VGG(pred, GT)
```

---

## 📈 Hasil Evaluasi

| Model | PSNR (dB) ↑ | SSIM ↑ |
|-------|:-----------:|:------:|
| Combo 4 — MobileViT Student | 40.94 | 0.9261 |
| **Combo 5 — Tiny-GAN Generator** | **43.85** | **0.9640** |

> 🏆 Combo 5 unggul di kedua metrik — PSNR lebih tinggi **+2.91 dB** dan SSIM lebih tinggi **+0.038** dibanding Combo 4, berkat adversarial training dan Sobel edge loss yang mempertahankan ketajaman objek.

**Interpretasi:**
- PSNR > 40 dB → kualitas restorasi **Excellent**
- SSIM > 0.92 → kemiripan struktural **Excellent**

---

## 🚀 Cara Menjalankan

### Prerequisites

```bash
# Platform: Kaggle Notebook (GPU T4 x2 recommended)
# Framework: TensorFlow 2.x
```

### Setup di Kaggle

1. **Add Dataset** → search `colorlabeilat seathru` → Add
2. **Accelerator** → GPU T4 x2
3. **Import Notebook** → upload file `.ipynb`

### Run Order

```python
# 1. Run preprocessing (Cell 1-15)
# 2. Run shared utilities (Cell 16-17)
# 3. Run Combo 4 training (Cell 18-22)
# 4. Run Combo 5 training (Cell 23-27)
# 5. Run comparison (Cell 28)
```

> ⚠️ `train_ds` menggunakan `.repeat()` → **WAJIB** set `TRAIN_STEPS` agar training tidak berjalan infinite!

```python
TRAIN_STEPS = 500   # steps per epoch
VAL_STEPS   = 100
EPOCHS      = 10
BATCH_SIZE  = 4     # turunkan jika OOM
```

---

## 📁 Struktur File

```
📦 project/
 ┣ 📓 full_notebook_combo4_5.ipynb   ← Notebook utama (Preprocessing + Combo 4 + 5)
 ┣ 📓 kombinasi5_only.ipynb          ← Notebook khusus Combo 5
 ┣ 📓 project_revised_fixed.ipynb    ← Notebook Combo 1 (Cheryl/teman)
 ┣ 📄 README.md
 ┗ 📂 /kaggle/working/
    ┣ best_student_combo4.weights.h5
    ┣ best_generator_combo5.weights.h5
    ┣ before_after_combo4.png
    ┣ before_after_combo5.png
    ┗ comparison_chart.png
```

---

## ⚠️ Tantangan & Keterbatasan

- **Tidak ada Ground Truth asli** → menggunakan Pseudo GT berbasis inversi fisika Sea-thru
- **OOM pada global attention** → diselesaikan dengan local attention patch 8×8
- **Teacher tidak pretrained** → keterbatasan komputasi (Kaggle free tier 30 jam/minggu)
- **Domain gap** → model dilatih di Red Sea, mungkin perlu fine-tuning untuk kondisi air lain

---

## 🔮 Future Work

- [ ] Pretrain Teacher di dataset UIEB/RUIE sebelum distilasi
- [ ] Collect paired GT dataset dengan kamera bawah air khusus
- [ ] Quantization (INT8) + ONNX export untuk deployment AUV
- [ ] Multi-domain training (Sea-thru + UIEB + EUVP)
- [ ] Extended training (50+ epoch, full steps)

---

## 📚 References

- Islam, M. J., et al. (2020). *Fast Underwater Image Enhancement for Improved Visual Perception*. IEEE RA-L.
- Yang, M., et al. (2015). *Underwater Image Enhancement Based on Conditional Generative Adversarial Network*. Signal Processing: Image Communication.
- Han, K., et al. (2020). *GhostNet: More Features from Cheap Operations*. CVPR.
- Mehta, S., et al. (2021). *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer*. ICLR.
- Akkaynak, D., & Treibitz, T. (2019). *Sea-thru: A Method for Removing Water From Underwater Images*. CVPR.

---

<div align="center">

Made with 🌊 by **Kelompok 3** — Binus University

*"From murky depths to crystal clarity"*

</div>

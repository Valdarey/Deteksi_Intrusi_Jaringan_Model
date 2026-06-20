# Deteksi Intrusi Jaringan — NSL-KDD

Project ini menggunakan **supervised learning (klasifikasi multi-kelas)** untuk mendeteksi dan mengklasifikasikan jenis serangan pada lalu lintas jaringan menggunakan dataset **NSL-KDD**.

## Live Demo

🔗 **Live Demo:** [deteksiintrusijaringanmodel.streamlit.app](https://deteksiintrusijaringanmodel-g9xfatnn9rzgjfqshrswrs.streamlit.app/)

## Tentang Dataset

Dataset berisi **125.973 record koneksi jaringan** dengan fitur:

| Fitur | Keterangan |
|---|---|
| `duration`, `src_bytes`, `dst_bytes`, dll | 38 fitur numerik statistik koneksi (durasi, jumlah byte, rate error, dll) |
| `protocol_type` | Jenis protokol (tcp, udp, icmp) |
| `service` | Jenis layanan jaringan (70 nilai, mis. http, ftp_data, smtp) |
| `flag` | Status koneksi (11 nilai, mis. SF, S0, REJ) |
| `label` | Label serangan asli (23 jenis spesifik, mis. `neptune`, `smurf`, `satan`) |
| `difficulty` | Skor tingkat kesulitan klasifikasi (dibuang, bukan fitur prediktif) |

## Tujuan

Menjawab pertanyaan: *"Apakah pola lalu lintas jaringan bisa digunakan untuk mengenali jenis serangan secara otomatis, termasuk serangan yang jarang terjadi?"*

23 label serangan asli disederhanakan menjadi **5 kategori** sesuai standar literatur keamanan jaringan:
- **normal** — lalu lintas sah
- **DoS** (Denial of Service) — membanjiri/melumpuhkan layanan
- **Probe** — pengintaian/pemindaian sebelum serangan
- **R2L** (Remote to Local) — penyerang luar mencoba akses lokal
- **U2R** (User to Root) — eskalasi dari pengguna biasa ke akses root

Karena dataset memiliki label, pendekatan yang digunakan adalah **klasifikasi multi-kelas (supervised learning)** — bukan clustering.

## Tools & Library

- Python, Pandas, NumPy
- Scikit-learn (`LabelEncoder`, `OneHotEncoder`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`)
- Matplotlib, Seaborn (visualisasi)
- Joblib (simpan model)
- Streamlit (dashboard interaktif)

## Metodologi

1. **Cek kualitas data** — pengecekan missing value dan duplikat
2. **Mapping label** — 23 label spesifik → 5 kategori serangan
3. **Encoding fitur** — `OneHotEncoder` untuk 3 fitur kategorikal (`protocol_type`, `service`, `flag`), fitur numerik dilewatkan langsung (*passthrough*)
4. **Split data** dengan `stratify` (80% latih, 20% uji) agar proporsi kelas minoritas tetap terjaga
5. **Training & perbandingan 3 model**: Decision Tree, Random Forest (`class_weight='balanced'`), Gradient Boosting
6. **Evaluasi** dengan Accuracy, F1-macro, dan **Recall-macro** sebagai metrik utama
7. **Pemilihan model terbaik** berdasarkan recall macro tertinggi
8. **Analisis feature importance** dan confusion matrix

## Hasil Klasifikasi

Distribusi data sangat **tidak seimbang**:

| Kategori | Jumlah | Persentase |
|---|---|---|
| normal | 67.343 | 53,46% |
| DoS | 45.927 | 36,46% |
| Probe | 11.656 | 9,25% |
| R2L | 995 | 0,79% |
| U2R | 52 | 0,04% |

| Model | Accuracy | F1-macro | Recall-macro |
|---|---|---|---|
| Decision Tree | 0,9981 | 0,9282 | 0,9525 |
| **Random Forest** | **0,9989** | **0,9737** | **0,9710** |
| Gradient Boosting | 0,9973 | 0,8809 | 0,8576 |

**Insight:** Accuracy semua model nyaris sama (>99,7%) karena dominasi kelas `normal`/`DoS`, tetapi **Random Forest paling unggul mengenali kelas minoritas** (U2R, R2L) berkat `class_weight='balanced'`, sehingga dipilih sebagai model terbaik. Fitur paling berpengaruh adalah `src_bytes`, `dst_bytes`, dan `dst_host_srv_count` — mencerminkan bahwa volume data dan pola perilaku terhadap host tujuan adalah sinyal kuat pembeda serangan.

## Cara Menjalankan

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook model_nsl.ipynb
```

## Dashboard Interaktif (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard mencakup:
- Ringkasan & kualitas data
- Distribusi serangan (skala log)
- Perbandingan metrik 3 model
- Confusion matrix per model
- Feature importance per model
- Uji prediksi (sampel tunggal & batch CSV)
- Unduh model terlatih (`model_nsl.pkl`)

---

**Dibuat sebagai project pembelajaran Data Science — Supervised Learning untuk Deteksi Intrusi Jaringan dengan NSL-KDD.**

# Sentiment Analysis: Ulasan Aplikasi Shopee (NLP)

Proyek **Natural Language Processing** untuk analisis sentimen ulasan aplikasi Shopee (Google Play). Menggunakan leksikon bahasa Indonesia (InSet) untuk pelabelan dan model BERT Indonesia (IndoBERT, IndoBERTweet, DistilIndoBERT) untuk klasifikasi biner: **positive** vs **negative**.

---


## Deskripsi singkat

- **Data:** Ulasan aplikasi Shopee dari Google Play (`shopee_mobile_reviews.csv`, ~40.000 baris).
- **Tujuan:** Klasifikasi sentimen (positive/negative) dengan model BERT berbahasa Indonesia.
- **Pipeline:** Load data → labeling sentimen (InSet) → filter & preprocessing teks → EDA (ngram, word cloud) → train/validation split → fine-tuning BERT → evaluasi & perbandingan model.

---

## Struktur proyek

```
.
├── [NLP]_MUHAMMAD_PHOHAN.ipynb   # Notebook utama: EDA + training + evaluasi
├── shopee_mobile_reviews.csv     # Data ulasan (harus ada untuk menjalankan notebook)
├── requirements.txt              # Dependency (opsional; notebook pakai pip install)
├── README.md                     # File ini
└── best_combo*_70_30/           # Folder model tersimpan (jika sudah di-train)
```



## Ringkasan kode dan alur

| Bagian | Isi |
|--------|-----|
| **Setup** | Install paket (numpy, pandas, transformers, datasets, scikit-learn, Sastrawi, wordcloud); import library. |
| **Data** | Load CSV; kolom utama: `content` (teks ulasan), `score` (bintang). ~40k baris. |
| **Labeling sentimen** | Lexicon InSet (positive/negative) dari GitHub; hitung skor per kalimat → label **positive** / **negative**. Baris tanpa label jelas di-drop. |
| **Preprocessing** | Lowercase, hapus URL, normalisasi elongasi, kamus normalisasi (gk→tidak, dll.), stopword (Sastrawi + custom). Kolom untuk BERT: `text_bert`; untuk EDA: `text_ng`, `text_uni` (stemming + stopword). Teks kosong dihapus → ~11.935 sampel. |
| **EDA** | Distribusi sentimen (countplot); top unigram/bigram/trigram global dan per sentimen (bar plot); WordCloud global, negatif, positif. |
| **Split** | Train/test 70/30; stratify menurut label. |
| **Training** | Fine-tune 3 model: **IndoBERT base**, **IndoBERTweet base**, **DistilIndoBERT**; 3 epoch; F1-macro untuk pilih checkpoint terbaik; model disimpan di `best_combo*_70_30`. |
| **Evaluasi** | Accuracy, F1 macro, F1 weighted; `classification_report`; confusion matrix (dan heatmap di notebook). |

---

## Hasil analisis dan output utama

### 1.  preprocessing

- **Jumlah sampel:** 11.935 (dengan teks tidak kosong dan label sentimen).
- **Distribusi sentimen (dari output notebook):**
  - **Negative:** 7.225
  - **Positive:** 4.669  

  (Imbalance sedang ke arah negatif.)

### 2. EDA – Kata dan n-gram yang menonjol

- **Top 5 unigram (global):** belanja (2693), kirim (2487), kalau (1856), iklan (1843), lama (1580).
- **Bigram:** mis. "sangat bantu", "tiba tiba", "customer service", "terima kasih", "jasa kirim".
- **Trigram:** mis. "tiba tiba masuk", "makin kesini makin", "sangat bantu belanja".
- **WordCloud:** Global, negatif, dan positif (visualisasi di notebook).

### 3. Hasil model (test set, 70/30)

| Run | Model | Accuracy | F1 (macro) | F1 (weighted) |
|-----|--------|----------|------------|----------------|
| combo1 | indolem/indobert-base-uncased | 94.96% | 94.67% | 94.94% |
| combo2 | indolem/indobertweet-base-uncased | 95.12% | 94.84% | 95.10% |
| **combo3** | **DistilIndoBERT** | **95.94%** | **95.71%** | **95.92%** |

**Model terbaik:** **combo3_distilindobert_70_30** (DistilIndoBERT).

### 4. Classification report (model terbaik, test set)

| Class    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| negative | 0.96      | 0.98   | 0.97     | 2168    |
| positive | 0.97      | 0.93   | 0.95     | 1401    |
| **accuracy** |         |        | **0.96** | **3569** |
| macro avg    | 0.96  | 0.95   | 0.96     | 3569    |
| weighted avg | 0.96  | 0.96   | 0.96     | 3569    |

### 5. Confusion matrix (model terbaik)

- **TN:** 2123 | **FP:** 45  
- **FN:** 100 | **TP:** 1301  


---

## Kesimpulan singkat

- **Data:** Ulasan Shopee berbahasa Indonesia dengan sentimen dari leksikon InSet; setelah preprocessing ~12k sampel, distribusi agak tidak seimbang (lebih banyak negatif).
- **EDA:** Kata seperti "belanja", "kirim", "iklan", "lama" mendominasi; bigram/trigram dan WordCloud menggambarkan tema pengiriman, iklan, dan pengalaman belanja.
- **Model:** Fine-tuning BERT Indonesia (IndoBERT, IndoBERTweet, DistilIndoBERT) menghasilkan accuracy ~95–96%; **DistilIndoBERT memberikan hasil terbaik** (accuracy 95.94%, F1 macro 95.71%) dengan model lebih ringan.
- **Output di notebook:** Grafik (countplot, bar n-gram, WordCloud, confusion matrix heatmap) dan tabel (classification report, ringkasan run) tersimpan sebagai output cell di `[NLP]_MUHAMMAD_PHOHAN.ipynb`.

---


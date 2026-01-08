# 🍳 RAG Tarif Arama Sistemi

**Derin Öğrenme Dersi - RAG + LLM Projesi**

Türk mutfağına ait ~20.000 tarifi içeren bir **Retrieval-Augmented Generation (RAG)** sistemi. Farklı embedding modelleri, chunking stratejileri ve LLM'ler kullanılarak karşılaştırmalı analiz yapılmıştır.

---

## 📋 İçindekiler

1. [Proje Yapısı](#1-proje-yapısı)
2. [Veri Hazırlığı](#2-veri-hazırlığı)
3. [Chunking Stratejileri](#3-chunking-stratejileri)
4. [Embedding Modelleri](#4-embedding-modelleri)
5. [Vektör Veritabanı](#5-vektör-veritabanı)
6. [Retriever-Only Sonuçları](#6-retriever-only-sonuçları)
7. [LLM-Only Sonuçları](#7-llm-only-sonuçları)
8. [RAG + LLM Sonuçları](#8-rag--llm-sonuçları)
9. [Final Comparison Table](#9-final-comparison-table)
10. [Tartışma (Zorunlu 5 Soru)](#10-tartışma-zorunlu-5-soru)
11. [Hallucination Örnekleri](#11-hallucination-örnekleri)
12. [Human Evaluation](#12-human-evaluation) 

---

## 1. Proje Yapısı

```
Proje proje/
├── 1- Veri Kazıma ve Temizleme/    # Web scraping
├── 2- bge-m3 Qdrant WholeDocument/ # BGE-M3 + WholeDoc
├── 3- e5-large Qdrant WholeDocument/ # E5-Large + WholeDoc
├── 4- bge-m3 Qdrant ParentChild/   # BGE-M3 + Parent-Child
├── 5- Retriever Evaluation/        # Retriever değerlendirme
├── 6- RAG Pipeline/                # RAG + LLM sistemi
├── 7- LLM Evaluation/              # LLM değerlendirme
├── requirements.txt
└── README.md
```

---

## 2. Veri Hazırlığı

| Özellik | Değer |
|---------|-------|
| **Kaynak** | yemek.com |
| **Toplam Tarif** | 20,554 |
| **Format** | JSONL |
| **İçerik** | Başlık, malzemeler, yapılış, URL |
| **Temizleme** | HTML artifacts, tekrar kayıtlar silindi |

---

## 3. Chunking Stratejileri

| Strateji | Açıklama | Chunk Sayısı |
|----------|----------|--------------|
| **WholeDocument** | Tüm tarif tek vektör | 20,554 |
| **Parent-Child** | Malzeme + Talimat ayrı | 41,108 |

---

## 4. Embedding Modelleri

| Model | Boyut | Encoding Time* | Özellik |
|-------|-------|----------------|---------|
| **BAAI/bge-m3** | 1024 | ~45ms/doc | Çok dilli, Türkçe güçlü |
| **intfloat/multilingual-e5-large** | 1024 | ~38ms/doc | Query/Passage prefix |

*Ortalama encoding süresi (CPU, batch=1)

---

## 5. Vektör Veritabanı

| Parametre | Değer |
|-----------|-------|
| **Veritabanı** | Qdrant (lokal) |
| **Benzerlik Metriği** | Cosine |
| **Index Tipi** | HNSW |
| **k Değerleri** | 1, 3, 5, 10 |

---

## 6. Retriever-Only Sonuçları

**Evaluation Set:** 60 soru (50 normal + 10 impossible)

### Retriever Performans Tablosu (k=5)

| Sistem | Recall@5 | Hit Rate@5 | MRR@5 | Latency |
|--------|----------|------------|-------|---------|
| **BGE-M3 Parent-Child** | **17.10%** | **46.00%** | 0.340 | 811ms |
| BGE-M3 WholeDocument | 16.20% | 44.00% | **0.366** | **598ms** |
| E5-Large WholeDocument | 13.40% | 38.00% | 0.311 | 662ms |

### k Değerleri Karşılaştırması (Hit Rate)

| Sistem | k=1 | k=3 | k=5 | k=10 |
|--------|-----|-----|-----|------|
| BGE-M3 WholeDocument | 32% | 40% | 44% | 56% |
| E5-Large WholeDocument | 28% | 32% | 38% | 52% |
| BGE-M3 Parent-Child | 26% | 42% | 46% | **60%** |

> **Not:** False Positive Rate tüm sistemlerde %100. Impossible sorulara yüksek benzerlik skoru veriliyor - bu sorun LLM katmanında çözülmektedir.

---

## 7. LLM-Only Sonuçları

**Desteklenen LLM'ler:**
- **API:** OpenAI GPT-4o-mini, Groq Llama 3.3 70B
- **Lokal (Ollama):** Qwen2 1.5B, Llama 3.2 3B, Phi-3 Mini, Mistral 7B

### LLM-Only Performans

| Model | Tip | F1 | Combined | Hallucination | Latency |
|-------|-----|-----|----------|---------------|---------|
| OpenAI GPT-4o-mini | API | 20.95% | 34.45% | 0% | 4.4s |
| Groq Llama 3.3 70B | API | 18.50% | 31.75% | 10% | 2.1s |
| Qwen2 1.5B | Lokal | 10.29% | 15.69% | **50%** | 15s |
| Llama 3.2 3B | Lokal | 14.70% | 22.57% | 20% | 39s |
| Phi-3 Mini | Lokal | 8.50% | 11.85% | 40% | 28s |
| Mistral 7B | Lokal | 12.30% | 18.45% | 30% | 42s |

---

## 8. RAG + LLM Sonuçları

### RAG Performans

| Model | Tip | F1 | Combined | Hallucination | Latency |
|-------|-----|-----|----------|---------------|---------|
| OpenAI GPT-4o-mini | API | 14.03% | **42.99%** | 0% | 7.7s |
| Groq Llama 3.3 70B | API | 15.20% | **41.75%** | 0% | 5.3s |
| Qwen2 1.5B | Lokal | 12.32% | **36.57%** | 0% | 34s |
| Llama 3.2 3B | Lokal | 12.19% | **39.52%** | 10% | 55s |
| Phi-3 Mini | Lokal | 10.10% | **32.75%** | 10% | 48s |
| Mistral 7B | Lokal | 11.85% | **35.20%** | 5% | 58s |

### RAG İyileştirme Oranları

| Model | LLM-Only | RAG | İyileşme | Hall. Azalma |
|-------|----------|-----|----------|--------------|
| OpenAI GPT-4o-mini | 34.45% | 42.99% | **+24.8%** | - |
| Groq Llama 3.3 70B | 31.75% | 41.75% | **+31.5%** | %100 |
| Qwen2 1.5B | 15.69% | 36.57% | **+133.1%** | **%100** |
| Llama 3.2 3B | 22.57% | 39.52% | **+75.1%** | %50 |
| Phi-3 Mini | 11.85% | 32.75% | **+176.4%** | %75 |
| Mistral 7B | 18.45% | 35.20% | **+90.8%** | %83 |

---

## 9. Final Comparison Table

**PDF Section 7: Zorunlu Karşılaştırma Tablosu (3 Sistem)**

| SYSTEM | EM ↑ | F1 ↑ | Human Relevance ↑ | Faithfulness ↑ | Hallucination ↓ | Latency ↓ |
|--------|------|------|-------------------|----------------|-----------------|-----------|
| **Retriever-Only** | – | – | – | – | – | **598ms** (fastest) |
| **LLM-Only** | 0% | 14.21% | 3.20 | 2.64 | **32%** (HIGH) | 24.6s |
| **RAG + LLM** | 0% | 12.53% | **4.53** | **4.87** | **6%** (LOW) | 39.5s |

> **Not:** Ortalama değerler (6 model üzerinden). RAG+LLM en yüksek Human scores ve en düşük Hallucination.

### Model Bazlı Detay

| Model | Mode | EM | F1 | Relevance | Faithfulness | Hall. | Latency |
|-------|------|-----|-----|-----------|--------------|-------|---------|
| OpenAI GPT-4o-mini | LLM-Only | 0% | 20.95% | 4.20 | 3.60 | 0% | 4.4s |
| OpenAI GPT-4o-mini | RAG | 0% | 14.03% | **5.00** | **5.00** | 0% | 7.7s |
| Qwen2 1.5B | LLM-Only | 0% | 10.29% | 2.20 | 1.60 | **50%** | 15s |
| Qwen2 1.5B | RAG | 0% | 12.32% | 4.00 | 4.60 | **0%** | 34s |
| Llama 3.2 3B | LLM-Only | 0% | 14.70% | 3.20 | 2.40 | 20% | 39s |
| Llama 3.2 3B | RAG | 0% | 12.19% | 4.60 | 5.00 | 10% | 55s |
| Mistral 7B | LLM-Only | 0% | 12.30% | 3.40 | 2.80 | 30% | 42s |
| Mistral 7B | RAG | 0% | 11.85% | 4.40 | 4.80 | 5% | 58s |

---

## 10. Tartışma (Zorunlu 5 Soru)

### 1. RAG retriever başarılı mı?

**Kısmen başarılı.** Hit Rate@10'da %60 başarı sağlandı. Ancak impossible sorulara da yüksek skor veriliyor (FP Rate %100). Bu sorun LLM katmanında context doğrulama ile çözüldü.

### 2. LLM-only neden düşük performanslıdır?

| Problem | Açıklama |
|---------|----------|
| Hallucination | Küçük modellerde %50'ye varan oran |
| Domain Bilgisi | Türk mutfağı detaylarını bilmiyorlar |
| Tutarsızlık | Her sorguda farklı cevap |

### 3. RAG sistemi ne kadar iyileştirme sağladı?

| Model | İyileşme | Hall. Azalma |
|-------|----------|--------------|
| OpenAI | +24.8% | - |
| Qwen2 | **+133.1%** | **%100** |
| Llama 3.2 | +75.1% | %50 |
| Phi-3 | +176.4% | %75 |

**Ortalama iyileştirme: %78**

### 4. Lokal vs API LLM farkı?

| Kriter | API (OpenAI/Groq) | Lokal (Ollama) |
|--------|-------------------|----------------|
| Kalite | Yüksek (42.99%) | Orta (32-39%) |
| Hız | 2-8s | 15-55s |
| Türkçe | Mükemmel | Zayıf-Orta |
| Gizlilik | Veri dışarı | Yerel |

### 5. En ideal konfigürasyon?

**BGE-M3 WholeDocument + OpenAI GPT-4o-mini**
- Combined Score: 42.99%
- Hallucination: 0%
- Human Avg: 5.00/5.00
- Latency: 7.7s

---

## 11. Hallucination Örnekleri

### LLM-Only Halüsinasyonları

| Soru | LLM Cevabı | Gerçek |
|------|------------|--------|
| "Dondurmalı karnıyarık" | "Üzerine dondurma konur..." | **Mevcut değil** |
| "Çikolatalı mercimek çorbası" | "Kakao eklenir..." | **Absürt** |

### RAG ile Düzeltme

| Soru | RAG Cevabı |
|------|------------|
| "Dondurmalı karnıyarık" | "Bu tarif veritabanında bulunmamaktadır." |

### Hallucination Oranları

| Model | LLM-Only | RAG | Azalma |
|-------|----------|-----|--------|
| OpenAI GPT-4o | 0% | 0% | - |
| Groq Llama 3.3 | 10% | 0% | %100 |
| Qwen2 1.5B | 50% | 0% | **%100** |
| Llama 3.2 3B | 20% | 10% | %50 |
| Phi-3 Mini | 40% | 10% | %75 |
| Mistral 7B | 30% | 5% | %83 |

---

## 12. Human Evaluation

**Değerlendirici:** Emircan Demir  
**Örnek:** 10 soru × 6 model × 2 mod = 120 değerlendirme

### Kriter Açıklamaları

| Kriter | Açıklama |
|--------|----------|
| **Relevance** | Cevap soruyla ilgili mi? (1-5) |
| **Faithfulness** | Context'e sadık mı? (1-5) |
| **Fluency** | Türkçe akıcı mı? (1-5) |

### Özet Sonuçlar

| Model | Mode | Relevance | Faithfulness | Fluency | **Avg** |
|-------|------|-----------|--------------|---------|---------|
| OpenAI GPT-4o | LLM-Only | 4.20 | 3.60 | 5.00 | 4.27 |
| OpenAI GPT-4o | RAG | **5.00** | **5.00** | **5.00** | **5.00** |
| Groq Llama 3.3 | LLM-Only | 4.00 | 3.40 | 4.80 | 4.07 |
| Groq Llama 3.3 | RAG | 4.80 | 4.80 | 4.80 | 4.80 |
| Qwen2 1.5B | LLM-Only | 2.20 | 1.60 | 3.00 | 2.27 |
| Qwen2 1.5B | RAG | 4.00 | 4.60 | 4.00 | 4.20 |
| Llama 3.2 3B | LLM-Only | 3.20 | 2.40 | 4.00 | 3.20 |
| Llama 3.2 3B | RAG | 4.60 | 5.00 | 4.00 | 4.53 |
| Phi-3 Mini | LLM-Only | 2.40 | 1.80 | 3.20 | 2.47 |
| Phi-3 Mini | RAG | 4.20 | 4.40 | 3.80 | 4.13 |
| Mistral 7B | LLM-Only | 3.40 | 2.80 | 3.60 | 3.27 |
| Mistral 7B | RAG | 4.40 | 4.80 | 3.80 | 4.33 |

### Ortalama Karşılaştırma (Tüm Modeller)

| Mode | Relevance | Faithfulness | Fluency | **Ortalama** |
|------|-----------|--------------|---------|--------------|
| LLM-Only | 3.23 | 2.60 | 3.93 | 3.26 |
| RAG | **4.50** | **4.77** | **4.23** | **4.50** |
| **Fark** | +1.27 | **+2.17** | +0.30 | **+1.24** |

**Bulgular:**
- RAG her modelde ve kriterde LLM-Only'den üstün
- En büyük iyileşme: **Faithfulness +2.17 puan** (context sadakati)
- Küçük modellerde (Qwen2, Phi-3) fark daha belirgin
- Türkçe akıcılık (Fluency) API modellerinde daha yüksek

---

## 👨‍💻 Geliştirici

**Emircan Demir** - [@EmircanDemirTR](https://github.com/EmircanDemirTR)
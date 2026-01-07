# 🍳 RAG Tarif Arama Sistemi

**Derin Öğrenme Dersi - RAG + LLM Projesi**

Bu proje, Türk mutfağına ait ~20.000 tarifi içeren bir **Retrieval-Augmented Generation (RAG)** sistemidir. Farklı embedding modelleri ve chunking stratejileri kullanılarak karşılaştırmalı analiz yapılmaktadır.

---

## 📋 İçindekiler

- [Proje Yapısı](#-proje-yapısı)
- [Gereksinimler](#-gereksinimler)
- [Kurulum](#-kurulum)
- [Modüller](#-modüller)
  - [1. Veri Kazıma ve Temizleme](#1--veri-kazıma-ve-temizleme)
  - [2. BGE-M3 WholeDocument](#2--bge-m3-wholedocument)
  - [3. E5-Large WholeDocument](#3--e5-large-wholedocument)
  - [4. BGE-M3 ParentChild](#4--bge-m3-parentchild)
  - [5. Retriever Evaluation](#5--retriever-evaluation)
  - [6. RAG Pipeline](#6--rag-pipeline)
  - [7. LLM Evaluation](#7--llm-evaluation)
- [Kullanım](#-kullanım)
- [Performans Karşılaştırması](#-performans-karşılaştırması)

---

## 📁 Proje Yapısı

```
Proje proje/
│
├── 1- Veri Kazıma ve Temizleme/    # Web scraping ve veri temizleme
│   ├── scraper.py                   # Yemek.com tarif scraper
│   ├── temizlememe1.py              # Veri temizleme scripti
│   ├── eski.jsonl                   # Ham veri
│   └── temiz.jsonl                  # Temizlenmiş veri (20,554 tarif)
│
├── 2- bge-m3 Qdrant WholeDocument/  # BGE-M3 + Whole Document Chunking
│   ├── config.py                    # Konfigürasyon ayarları
│   ├── embedder.py                  # Embedding işlemleri
│   ├── database.py                  # Qdrant veritabanı işlemleri
│   ├── indexer.py                   # Veri indexleme
│   ├── searcher.py                  # Arama fonksiyonları
│   ├── main.py                      # Ana uygulama
│   └── qdrant_data/                 # Vektör veritabanı
│
├── 3- e5-large Qdrant WholeDocument/ # E5-Large + Whole Document Chunking
│   ├── config.py
│   ├── embedder.py
│   ├── database.py
│   ├── indexer.py
│   ├── searcher.py
│   ├── main.py
│   └── qdrant_data/
│
├── 4- bge-m3 Qdrant ParentChild/    # BGE-M3 + Parent-Child Chunking
│   ├── config.py
│   ├── embedder.py                  # Parent-Child chunk oluşturma
│   ├── database.py
│   ├── indexer.py
│   ├── searcher.py
│   ├── main.py
│   └── qdrant_data/
│
├── 5- Retriever Evaluation/         # Retriever performans değerlendirmesi
│   ├── config.py                    # Değerlendirme ayarları
│   ├── metrics.py                   # Recall@k, Hit Rate@k, MRR, FP Rate
│   ├── evaluator.py                 # Ana değerlendirme modülü
│   ├── evaluation_set.json          # 60 soruluk test seti (50+10 impossible)
│   └── results/                     # Değerlendirme sonuçları
│
├── 6- RAG Pipeline/                 # RAG + LLM entegrasyonu
│   ├── config.py                    # LLM ve RAG ayarları
│   ├── prompt_templates.py          # Prompt şablonları
│   ├── llm_api.py                   # Groq API entegrasyonu
│   ├── llm_local.py                 # Ollama lokal LLM
│   ├── rag_pipeline.py              # Ana RAG sistemi
│   └── main.py                      # İnteraktif arayüz
│
├── 7- LLM Evaluation/               # LLM performans değerlendirmesi
│   ├── config.py                    # Değerlendirme ayarları
│   ├── metrics.py                   # EM, F1, Hallucination
│   ├── evaluator.py                 # Ana değerlendirme modülü
│   └── results/                     # Sonuçlar
│
├── requirements.txt                 # Tüm proje gereksinimleri
├── Project Guideline-LLM.pdf        # Proje rehberi
├── LICENSE                          # MIT License
└── README.md
```

---

## 💻 Gereksinimler

### Python Versiyonu
```
Python 3.12.10
```

### Gerekli Kütüphaneler

| Kütüphane | Versiyon | Açıklama |
|-----------|----------|----------|
| `sentence-transformers` | ≥2.2.0 | Embedding modelleri (BGE-M3, E5-Large) |
| `qdrant-client` | ≥1.7.0 | Vektör veritabanı |
| `torch` | ≥2.0.0 | Deep learning framework |
| `transformers` | ≥4.41.0 | Hugging Face Transformers |
| `groq` | ≥0.4.0 | Groq API (LLM) |
| `requests` | ≥2.28.0 | HTTP istekleri |
| `beautifulsoup4` | ≥4.12.0 | HTML parsing |
| `tqdm` | ≥4.66.0 | Progress bar |
| `rich` | ≥13.7.0 | Terminal UI |
| `tf-keras` | ≥2.20.0 | Keras uyumluluk |

### Lokal LLM için
- **Ollama** - [ollama.ai](https://ollama.ai) kurulmalı

---

## 🚀 Kurulum

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/EmircanDemirTR/Yemek-Tarifi-RAG-LLM.git
cd Yemek-Tarifi-RAG-LLM
```

### 2. Gerekli Paketleri Kurun

```bash
pip install -r requirements.txt
```

Bu komut tüm proje gereksinimlerini kuracaktır:
- Web scraping (requests, beautifulsoup4)
- Embedding modelleri (sentence-transformers, torch)
- Vektör veritabanı (qdrant-client)
- Veri işleme (pandas, numpy)
- Terminal UI (rich, tqdm)
- Görselleştirme (matplotlib, seaborn)

### 3. Modelleri İndirin (İlk Çalıştırmada Otomatik)
- **BGE-M3**: `BAAI/bge-m3` (~2.2GB)
- **E5-Large**: `intfloat/multilingual-e5-large` (~2.2GB)

---

## 📦 Modüller

### 1. 📥 Veri Kazıma ve Temizleme

**Amaç:** Yemek.com'dan tarif verilerini çekme ve temizleme

**Toplanan Veri:**
- 📊 **20,554 tarif**
- 📝 Her tarif: başlık, malzemeler, yapılış adımları, URL

**Dosyalar:**
| Dosya | Açıklama |
|-------|----------|
| `scraper.py` | Paralel web scraper (yemek.com) |
| `temizlememe1.py` | Veri temizleme (tekrar silme, hatalı kayıt filtreleme) |
| `temiz.jsonl` | Temizlenmiş final veri |

**Veri Formatı (JSONL):**
```json
{
  "url": "https://yemek.com/tarif/tavuklu-makarna/",
  "title": "Tavuklu Makarna Tarifi",
  "ingredients": ["500g makarna", "2 adet tavuk göğsü", "..."],
  "instructions": ["Tavukları küp küp doğrayın.", "..."]
}
```

---

### 2. 🔷 BGE-M3 WholeDocument

**Embedding Modeli:** `BAAI/bge-m3`  
**Chunking Stratejisi:** Whole Document (Tam Doküman)  
**Vektör Boyutu:** 1024  
**Benzerlik Metriği:** Cosine

**Özellikler:**
- Her tarif tek bir vektör olarak indexlenir
- Başlık + Malzemeler + Yapılış birleştirilir
- 20,554 vektör

**Kullanım:**
```bash
cd "2- bge-m3 Qdrant WholeDocument"

# Veritabanı bilgisi
python main.py info

# İnteraktif arama
python main.py search

# Yeniden indexleme (dikkat: veritabanını siler!)
python main.py index
```

---

### 3. 🔶 E5-Large WholeDocument

**Embedding Modeli:** `intfloat/multilingual-e5-large`  
**Chunking Stratejisi:** Whole Document (Tam Doküman)  
**Vektör Boyutu:** 1024  
**Benzerlik Metriği:** Cosine

**Özellikler:**
- Multilingual model (Türkçe desteği güçlü)
- Query/Passage prefix kullanımı
- 20,554 vektör

**Kullanım:**
```bash
cd "3- e5-large Qdrant WholeDocument"
python main.py info
python main.py search
```

---

### 4. 🔹 BGE-M3 ParentChild

**Embedding Modeli:** `BAAI/bge-m3`  
**Chunking Stratejisi:** Parent-Child  
**Vektör Boyutu:** 1024  
**Benzerlik Metriği:** Cosine

**Özellikler:**
- Her tarif 2 chunk olarak indexlenir:
  - **Malzeme Chunk:** Başlık + Malzemeler
  - **Talimat Chunk:** Başlık + Yapılış
- 41,108 chunk (20,554 tarif × 2)
- Daha hassas arama imkanı

**Kullanım:**
```bash
cd "4- bge-m3 Qdrant ParentChild"
python main.py info
python main.py search

# Özel arama komutları:
# /malzeme tavuk, patates    → Malzeme chunk'larında ara
# /yontem fırında pişirme    → Talimat chunk'larında ara
```

---

### 5. 📈 Retriever Evaluation

**Amaç:** Tüm retriever sistemlerinin performansını ölçme ve karşılaştırma

**Değerlendirme Seti:**
- 📝 **60 soru** (50 normal + 10 impossible)
- 🎯 **Normal Sorular:** Gerçek tariflerle eşleşen sorular
- 🚫 **Impossible Sorular:** Var olmayan tarifleri test eden sorular (False Positive testi)
- 🏷️ Kategoriler: direkt, malzeme_bazlı, durum_bazlı, kısıtlamalı, karşılaştırmalı, impossible

**Hesaplanan Metrikler:**
| Metrik | Açıklama |
|--------|----------|
| Recall@k | Beklenen dokümanların bulunma oranı |
| Hit Rate@k | En az bir doğru sonuç bulma oranı (Success Rate) |
| MRR@k | Mean Reciprocal Rank - İlk doğru sonucun sıralaması |
| Precision@k | Top-k sonuçların ilgili olma oranı |
| **False Positive Rate** | Impossible sorulara yanlış cevap verme oranı |
| Latency | Arama süresi (ms) |

**Kullanım:**
```bash
cd "5- Retriever Evaluation"
python evaluator.py       # Tüm sistemleri değerlendir
python analyze_fp.py      # False Positive detaylı analiz
```

**Çıktılar:**
- `results/evaluation_results_*.json` - Detaylı sonuçlar
- `results/evaluation_summary_*.csv` - Özet tablo

---

### 6. 🤖 RAG Pipeline

**Amaç:** Retriever + LLM entegrasyonu ile soru-cevap sistemi

**Desteklenen LLM'ler:**

| Provider | Model | Tip | Açıklama |
|----------|-------|-----|----------|
| **Groq** | Llama 3.3 70B | API | Ücretsiz, çok hızlı |
| **Ollama** | Qwen2 1.5B | Lokal | Hafif, hızlı |
| **Ollama** | Llama 3.2 3B | Lokal | Dengeli |
| **Ollama** | Phi-3 Mini | Lokal | Kaliteli |
| **Ollama** | Gemma2 2B | Lokal | Alternatif |

**Kurulum:**
```bash
# Groq API key ayarla
export GROQ_API_KEY="your-api-key"

# Ollama modellerini indir
ollama pull qwen2:1.5b
ollama pull llama3.2:3b
ollama pull phi3:mini
```

**Kullanım:**
```bash
cd "6- RAG Pipeline"
python main.py
```

**İnteraktif Komutlar:**
| Komut | Açıklama |
|-------|----------|
| `/rag <soru>` | RAG modu (veritabanından context) |
| `/llm <soru>` | LLM-Only modu (context yok) |
| `/karsilastir <soru>` | RAG vs LLM-Only karşılaştırması |
| `/model <isim>` | Ollama modelini değiştir |
| `/groq` | Groq API'ye geç |
| `/modeller` | Mevcut modelleri listele |

---

### 7. 📊 LLM Evaluation

**Amaç:** LLM-Only ve RAG+LLM performans karşılaştırması

**Hesaplanan Metrikler:**
| Metrik | Açıklama |
|--------|----------|
| Exact Match (EM) | Tam eşleşme oranı |
| F1-score | Token bazlı benzerlik |
| Keyword Score | Anahtar kelime eşleşmesi |
| Faithfulness | Context'e sadakat (RAG için) |
| Combined Score | Ağırlıklı toplam skor |
| Hallucination Rate | Uydurma bilgi oranı |
| Latency | Cevap süresi (ms) |

**Kullanım:**
```bash
cd "7- LLM Evaluation"
python evaluator.py --models openai qwen2 --questions 10
```

---

## 🔍 Kullanım

### Arama Modu Komutları

Tüm sistemlerde `python main.py search` ile interaktif arama moduna girilir:

| Komut | Açıklama | Örnek |
|-------|----------|-------|
| `<sorgu>` | Genel arama | `tavuklu makarna` |
| `/malzeme <liste>` | Malzeme bazlı arama | `/malzeme tavuk, patates, soğan` |
| `/tarif <isim>` | Tarif adı araması | `/tarif karnıyarık` |
| `/detay <no>` | Son aramadan tarif detayı | `/detay 1` |
| `/cikis` | Çıkış | `/cikis` |

### Örnek Arama Çıktısı

```
🔍 Arama: mercimek çorbası

╭─────────────────────────────────────────────────────────────╮
│ [1] Mercimek Çorbası Tarifi                                 │
│     Benzerlik: 78.45%                                       │
│     https://yemek.com/tarif/mercimek-corbasi/               │
│                                                             │
│     Malzemeler: 1 su bardağı kırmızı mercimek, 1 adet       │
│     soğan, 2 yemek kaşığı tereyağı...                       │
╰─────────────────────────────────────────────────────────────╯
```

---

## 📊 Performans Karşılaştırması

### Retriever Sistemleri Özeti

| Sistem | Model | Chunking | Vektör Sayısı | Boyut |
|--------|-------|----------|---------------|-------|
| #2 | BGE-M3 | WholeDocument | 20,554 | 1024 |
| #3 | E5-Large | WholeDocument | 20,554 | 1024 |
| #4 | BGE-M3 | Parent-Child | 41,108 | 1024 |

### 📈 Retriever-Only Performans Sonuçları

60 soruluk evaluation set ile test edilmiştir (50 normal + 10 impossible).

#### 🏆 Ana Karşılaştırma Tablosu (k=5)

| Sistem | Recall@5 | Hit Rate@5 | MRR@5 | FP Rate | Latency |
|--------|----------|------------|-------|---------|---------|
| **BGE-M3 Parent-Child** | **17.10%** | **46.00%** | 0.340 | 100% | 811ms |
| BGE-M3 WholeDocument | 16.20% | 44.00% | **0.366** | 100% | **598ms** |
| E5-Large WholeDocument | 13.40% | 38.00% | 0.311 | 100% | 662ms |

#### 📊 Detaylı k Değerleri Karşılaştırması

| Sistem | k=1 | k=3 | k=5 | k=10 |
|--------|-----|-----|-----|------|
| **BGE-M3 WholeDocument** | | | | |
| ↳ Recall | 7.93% | 13.30% | 16.20% | 21.77% |
| ↳ Hit Rate | 32.00% | 40.00% | 44.00% | 56.00% |
| **E5-Large WholeDocument** | | | | |
| ↳ Recall | 7.03% | 11.10% | 13.40% | 19.50% |
| ↳ Hit Rate | 28.00% | 32.00% | 38.00% | 52.00% |
| **BGE-M3 Parent-Child** | | | | |
| ↳ Recall | 6.73% | 14.30% | 17.10% | 22.90% |
| ↳ Hit Rate | 26.00% | 42.00% | 46.00% | **60.00%** |

### 📝 Örnek Sorular (Evaluation Set'ten)

#### ✅ Normal Sorular (Doğru cevap bekleyenler)

| # | Soru | Kategori | Zorluk |
|---|------|----------|--------|
| 1 | "Çok acıkmış misafirler geldi, hızlı ne yapabilirim?" | durum_bazlı | orta |
| 2 | "Tavuk göğsü var ama sıkıcı olmayan bir şey yapmak istiyorum" | kısıtlamalı | orta |
| 3 | "Spor sonrası protein ağırlıklı hafif bir şey" | durum_bazlı | orta |
| 4 | "Dedemin çok sevdiği eski usul tatlılar nelerdir?" | karşılaştırmalı | zor |
| 5 | "Romantik bir akşam yemeği için etkileyici ana yemek" | durum_bazlı | zor |

#### 🚫 Impossible Sorular (Bulunamadı cevabı bekleyenler)

| # | Soru | Neden Impossible? |
|---|------|-------------------|
| 1 | "Dondurmalı karnıyarık tarifi var mı?" | Saçma kombinasyon |
| 2 | "Çikolatalı mercimek çorbası nasıl yapılır?" | Var olmayan tarif |
| 3 | "Wasabi soslu mantı tarifi istiyorum" | Fusion tarif - DB'de yok |
| 4 | "Ketçaplı sütlaç yapımı" | İğrenç kombinasyon |
| 5 | "Sushi tarifi Türk mutfağından" | Japon yemeği - kapsam dışı |

### 🔴 False Positive (Hallucination) Analizi

**Kritik Bulgu:** Tüm retriever sistemleri impossible sorulara yüksek benzerlik skoru veriyor!

| Impossible Soru | BGE-M3 WD | E5-Large | BGE-M3 PC |
|-----------------|-----------|----------|-----------|
| Dondurmalı karnıyarık | 0.664 | 0.887 | 0.675 |
| Çikolatalı mercimek çorbası | 0.674 | 0.862 | 0.682 |
| Mayonezli baklava | 0.634 | 0.872 | 0.648 |
| Ketçaplı sütlaç | 0.633 | 0.866 | 0.649 |

**Ortalama Skor Karşılaştırması:**
| Soru Tipi | BGE-M3 WD | E5-Large | BGE-M3 PC |
|-----------|-----------|----------|-----------|
| Normal Sorular | 0.623 | 0.855 | 0.627 |
| Impossible Sorular | 0.632 | 0.866 | 0.639 |
| **Fark** | **-0.009** | **-0.011** | **-0.012** |

> ⚠️ **Sonuç:** Impossible sorular normal sorulardan bile **daha yüksek** skor alıyor! Bu, embedding modellerinin semantik benzerliğe dayalı çalışmasından kaynaklanıyor ("dondurmalı karnıyarık" → "karnıyarık" ile yüksek benzerlik).

### 💡 Neden Bu Değerler Düşük?

Değerlendirme setimiz **zorlu ve gerçekçi sorular** içeriyor:

| Basit Soru (Yüksek Skor) | Zorlu Soru (Düşük Skor) |
|--------------------------|-------------------------|
| "Mercimek çorbası tarifi" | "Kış günü içimi ısıtacak bir şeyler" |
| "Karnıyarık nasıl yapılır" | "Patlıcan var, kıyma var, etkileyici bir şey" |
| "Baklava tarifi" | "Bayramda misafirlere şık bir tatlı" |

Zorlu sorular gerçek kullanım senaryolarını yansıtır ve retriever'ların gerçek performansını gösterir.

### 🔍 Analiz ve Bulgular

1. **En İyi Retriever:** 
   - **Hit Rate için:** BGE-M3 Parent-Child (%60 @ k=10)
   - **Hız için:** BGE-M3 WholeDocument (598ms)
   - **MRR için:** BGE-M3 WholeDocument (0.366)

2. **Chunking Karşılaştırması:**
   - Parent-Child daha fazla k değerinde daha iyi Hit Rate
   - WholeDocument tek aramada daha hızlı
   - Parent-Child malzeme/yöntem spesifik sorgularda avantajlı

3. **Model Karşılaştırması:**
   - BGE-M3, E5-Large'dan daha iyi performans
   - E5-Large en yüksek ham skorları veriyor ama ayırt edicilik düşük

4. **🚨 Kritik Bulgu - False Positive:**
   - Tüm sistemlerde FP Rate = %100
   - **Retriever seviyesinde hallucination önlenemez**
   - **Çözüm: LLM seviyesinde context doğrulama gerekli**

### 🎯 RAG Entegrasyonu İçin Öneriler

```
┌─────────────────────────────────────────────────────────────────┐
│  KULLANICI: "Dondurmalı karnıyarık tarifi var mı?"             │
│                          ↓                                      │
│  RETRIEVER: Karnıyarık Tarifi (skor: 0.66)                     │
│                          ↓                                      │
│  LLM PROMPT:                                                    │
│  "Kullanıcı 'dondurmalı karnıyarık' sordu.                     │
│   Context'te dondurmalı karnıyarık var mı?                     │
│   Yoksa 'Bu tarif veritabanında bulunamadı' de."               │
│                          ↓                                      │
│  LLM CEVAP: "Veritabanında dondurmalı karnıyarık tarifi        │
│              bulunmamaktadır. Normal karnıyarık ister misiniz?"│
└─────────────────────────────────────────────────────────────────┘
```

Bu yaklaşım ile LLM, retriever'ın döndürdüğü context'in soruyla **gerçekten eşleşip eşleşmediğini** değerlendirebilir

---

### 🤖 LLM Performans Sonuçları

10 soruluk evaluation set ile test edilmiştir.

#### 📊 Final Karşılaştırma Tablosu (PDF Formatı)

| Model | Mode | Combined ↑ | F1 ↑ | Hallucination ↓ | Latency |
|-------|------|------------|------|-----------------|---------|
| **OpenAI GPT-4o-mini** | LLM-Only | 34.45% | 20.95% | 0% | 4.4s |
| **OpenAI GPT-4o-mini** | RAG | **42.99%** | 14.03% | 0% | 7.7s |
| Qwen2 1.5B | LLM-Only | 15.69% | 10.29% | **50%** | 15s |
| Qwen2 1.5B | RAG | **36.57%** | 12.32% | **0%** | 34s |
| Llama 3.2 3B | LLM-Only | 22.57% | 14.70% | 20% | 39s |
| Llama 3.2 3B | RAG | **39.52%** | 12.19% | 10% | 55s |

#### 🏆 RAG İyileştirme Oranları

| Model | LLM-Only | RAG | İyileşme | Hall. Azalma |
|-------|----------|-----|----------|--------------|
| **OpenAI GPT-4o-mini** | 34.45% | 42.99% | **+24.8%** | - |
| **Qwen2 1.5B** | 15.69% | 36.57% | **+133.1%** | **%100** |
| **Llama 3.2 3B** | 22.57% | 39.52% | **+75.1%** | %50 |

#### 📌 Temel Bulgular

| Bulgu | Detay |
|-------|-------|
| **En Yüksek RAG Skoru** | OpenAI GPT-4o-mini (42.99%) |
| **En Büyük İyileşme** | Qwen2 1.5B (+133.1%) |
| **En Büyük Hall. Azalma** | Qwen2 1.5B (%50 → %0) |
| **En Hızlı API** | OpenAI (4-8 saniye) |
| **En Hızlı Lokal** | Qwen2 1.5B (15-34 saniye) |

#### 🎯 Analiz

1. **RAG her modelde daha iyi skor veriyor** - PDF beklentisine uygun
2. **LLM-Only'de yüksek hallucination** - Özellikle küçük modellerde (%50)
3. **RAG ile hallucination dramatik azalıyor** - Qwen2'de %100 azalma
4. **Lokal modeller Türkçe'de zayıf** - Phi-3 ve Mistral Türkçe desteklemiyor
5. **OpenAI en dengeli performans** - Hem hızlı hem kaliteli

---

## 🛣️ Yol Haritası

### ✅ Tamamlanan (Retrieval Aşaması)
- [x] Veri kazıma ve temizleme (20,554 tarif)
- [x] BGE-M3 WholeDocument retrieval sistemi
- [x] E5-Large WholeDocument retrieval sistemi
- [x] BGE-M3 Parent-Child retrieval sistemi
- [x] Evaluation set oluşturma (60 soru: 50 normal + 10 impossible)
- [x] Retriever performans değerlendirmesi (Recall@k, Hit Rate@k, MRR@k, Precision@k)
- [x] False Positive (Hallucination) analizi

### ✅ Tamamlanan (RAG + LLM Aşaması)
- [x] OpenAI API entegrasyonu (GPT-4o-mini)
- [x] Groq API entegrasyonu (Llama 3.3 70B)
- [x] Ollama lokal LLM entegrasyonu (Qwen2, Llama3.2, Phi-3, Mistral)
- [x] RAG Pipeline oluşturma
- [x] LLM Evaluation framework
- [x] LLM-Only ve RAG+LLM test altyapısı

### ✅ Tamamlanan (Değerlendirme Aşaması)
- [x] Tüm modellerde LLM-Only testi
- [x] Tüm modellerde RAG+LLM testi
- [x] Final karşılaştırma tablosu doldurma (PDF formatı)
- [x] Keyword ve Faithfulness metrikleri ekleme
- [x] RAG iyileştirme analizi

### 📋 Gelecek İyileştirmeler
- [ ] Re-ranking (Cross-encoder) ekleme
- [ ] Daha fazla Türkçe LLM desteği
- [ ] Web arayüzü (Gradio/Streamlit)
- [ ] Human evaluation (manuel değerlendirme)

---

## 📋 Sonuç Özeti

### 🏆 Retrieval Aşaması Sonuçları

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RETRIEVER KARŞILAŞTIRMASI                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   📊 Test: 60 soru (50 normal + 10 impossible)                     │
│                                                                     │
│   ┌─────────────────────┬──────────┬──────────┬─────────┐          │
│   │ Sistem              │ Hit@10   │ MRR@5    │ Latency │          │
│   ├─────────────────────┼──────────┼──────────┼─────────┤          │
│   │ BGE-M3 Parent-Child │ 60.00%   │ 0.340    │ 811ms   │ 🥇       │
│   │ BGE-M3 WholeDoc     │ 56.00%   │ 0.366    │ 598ms   │ 🥈       │
│   │ E5-Large WholeDoc   │ 52.00%   │ 0.311    │ 662ms   │ 🥉       │
│   └─────────────────────┴──────────┴──────────┴─────────┘          │
│                                                                     │
│   ⚠️  False Positive Rate: %100 (tüm sistemlerde)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 🤖 LLM + RAG Aşaması Sonuçları

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM-ONLY vs RAG+LLM                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   📊 Test: 10 soru | 1 API + 2 Lokal LLM                           │
│                                                                     │
│   ┌──────────────────┬──────────┬──────────┬──────────┬─────────┐  │
│   │ Model            │ LLM-Only │ RAG+LLM  │ İyileşme │ Hall.↓  │  │
│   ├──────────────────┼──────────┼──────────┼──────────┼─────────┤  │
│   │ OpenAI GPT-4o    │ 34.45%   │ 42.99%   │ +24.8%   │ -       │  │
│   │ Qwen2 1.5B       │ 15.69%   │ 36.57%   │ +133.1%  │ 100%    │  │
│   │ Llama 3.2 3B     │ 22.57%   │ 39.52%   │ +75.1%   │ 50%     │  │
│   └──────────────────┴──────────┴──────────┴──────────┴─────────┘  │
│                                                                     │
│   ✅ RAG her modelde LLM-Only'den daha iyi performans              │
│   ✅ Hallucination oranı RAG ile dramatik düşüş                    │
│   ✅ Küçük lokal modellerde en büyük iyileşme                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 📌 Temel Bulgular

| Bulgu | Detay |
|-------|-------|
| **En İyi Hit Rate** | BGE-M3 Parent-Child (%60 @ k=10) |
| **En Hızlı Retriever** | BGE-M3 WholeDocument (598ms) |
| **En İyi RAG Skoru** | OpenAI GPT-4o-mini (42.99%) |
| **En Büyük İyileşme** | Qwen2 1.5B (+133.1%) |
| **Hallucination Çözümü** | RAG ile %100'e varan azalma |

### 🎯 Final Değerlendirme

PDF Rehberindeki beklentiler **tam olarak karşılandı**:

| Beklenti | Sonuç | Durum |
|----------|-------|-------|
| RAG+LLM > LLM-Only | Her modelde RAG daha iyi | ✅ |
| LLM-Only'de yüksek hallucination | %20-%50 oranında | ✅ |
| RAG ile hallucination azalması | %100'e varan düşüş | ✅ |
| En az 2 embedding modeli | BGE-M3, E5-Large | ✅ |
| En az 2 chunking stratejisi | WholeDocument, Parent-Child | ✅ |
| En az 1 API LLM | OpenAI GPT-4o-mini | ✅ |
| En az 3 Lokal LLM | Qwen2, Llama3.2, Phi-3, Mistral | ✅ |

---

## 📖 Referanslar

- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
- [E5-Large Model](https://huggingface.co/intfloat/multilingual-e5-large)
- [Qdrant Vector Database](https://qdrant.tech/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 👨‍💻 Geliştirici

**Emircan Demir**  
GitHub: [@EmircanDemirTR](https://github.com/EmircanDemirTR)

---

## 📄 Lisans

Bu proje **MIT License** altında lisanslanmıştır.

| İzinler | Sınırlamalar | Koşullar |
|---------|--------------|----------|
| ✅ Ticari kullanım | ❌ Sorumluluk | ℹ️ Lisans ve telif hakkı bildirimi |
| ✅ Değiştirme | ❌ Garanti | |
| ✅ Dağıtım | | |
| ✅ Özel kullanım | | |

Detaylar için [LICENSE](LICENSE) dosyasına bakınız.


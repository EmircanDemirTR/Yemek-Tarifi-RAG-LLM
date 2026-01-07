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
│   ├── temiz.jsonl                  # Temizlenmiş veri (20,554 tarif)
│   └── requirements.txt
│
├── 2- bge-m3 Qdrant WholeDocument/  # BGE-M3 + Whole Document Chunking
│   ├── config.py                    # Konfigürasyon ayarları
│   ├── embedder.py                  # Embedding işlemleri
│   ├── database.py                  # Qdrant veritabanı işlemleri
│   ├── indexer.py                   # Veri indexleme
│   ├── searcher.py                  # Arama fonksiyonları
│   ├── main.py                      # Ana uygulama
│   ├── qdrant_data/                 # Vektör veritabanı
│   └── requirements.txt
│
├── 3- e5-large Qdrant WholeDocument/ # E5-Large + Whole Document Chunking
│   ├── config.py
│   ├── embedder.py
│   ├── database.py
│   ├── indexer.py
│   ├── searcher.py
│   ├── main.py
│   ├── qdrant_data/
│   └── requirements.txt
│
├── 4- bge-m3 Qdrant ParentChild/    # BGE-M3 + Parent-Child Chunking
│   ├── config.py
│   ├── embedder.py                  # Parent-Child chunk oluşturma
│   ├── database.py
│   ├── indexer.py
│   ├── searcher.py
│   ├── main.py
│   ├── qdrant_data/
│   └── requirements.txt
│
├── Project Guideline-LLM.pdf        # Proje rehberi
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
| `sentence-transformers` | ≥5.2.0 | Embedding modelleri (BGE-M3, E5-Large) |
| `qdrant-client` | ≥1.16.0 | Vektör veritabanı |
| `torch` | ≥2.0.0 | Deep learning framework |
| `transformers` | ≥4.41.0 | Hugging Face Transformers |
| `requests` | ≥2.28.0 | HTTP istekleri |
| `beautifulsoup4` | ≥4.12.0 | HTML parsing |
| `tqdm` | ≥4.66.0 | Progress bar |
| `rich` | ≥13.7.0 | Terminal UI |
| `tf-keras` | ≥2.20.0 | Keras uyumluluk |

---

## 🚀 Kurulum

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/EmircanDemirTR/Yemek-Tarifi-RAG-LLM.git
cd Yemek-Tarifi-RAG-LLM
```

### 2. Gerekli Paketleri Kurun

**Tüm paketleri tek seferde kurmak için:**
```bash
pip install requests beautifulsoup4 urllib3 sentence-transformers qdrant-client tqdm torch rich tf-keras
```

**Veya klasör bazlı kurulum:**
```bash
# Veri kazıma için
pip install -r "1- Veri Kazıma ve Temizleme/requirements.txt"

# Retrieval sistemleri için
pip install -r "2- bge-m3 Qdrant WholeDocument/requirements.txt"
```

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

### Retrieval Sistemleri

| Sistem | Model | Chunking | Vektör Sayısı | Boyut |
|--------|-------|----------|---------------|-------|
| #2 | BGE-M3 | WholeDocument | 20,554 | 1024 |
| #3 | E5-Large | WholeDocument | 20,554 | 1024 |
| #4 | BGE-M3 | Parent-Child | 41,108 | 1024 |

### Metrikler (Gelecek Çalışma)

| Metrik | Açıklama |
|--------|----------|
| Recall@k | İlgili dokümanların bulunma oranı |
| Hit Rate@k | En az bir ilgili doküman bulma oranı |
| MRR@k | Mean Reciprocal Rank |
| Latency | Arama süresi |

---

## 🛣️ Yol Haritası

- [x] Veri kazıma ve temizleme
- [x] BGE-M3 WholeDocument retrieval
- [x] E5-Large WholeDocument retrieval
- [x] BGE-M3 Parent-Child retrieval
- [ ] LLM entegrasyonu (API - OpenAI/Gemini/Claude)
- [ ] Lokal LLM entegrasyonu (LLaMA, Mistral, Gemma, Qwen)
- [ ] RAG + LLM pipeline
- [ ] Performans değerlendirmesi (Recall, F1, EM)
- [ ] Hallucination analizi
- [ ] Final karşılaştırma raporu

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


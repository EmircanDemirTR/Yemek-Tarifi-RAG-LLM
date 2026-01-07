# -*- coding: utf-8 -*-
"""
🍳 RAG Tarif Arama Sistemi - Colab Indexer (E5-Large)
=====================================================
Bu script'i Google Colab'da L4 GPU ile çalıştırın.

Kullanım:
1. Colab'da yeni notebook açın
2. Runtime > Change runtime type > L4 GPU seçin
3. Bu dosyayı Colab'a yükleyin veya hücrelere kopyalayın
4. Çalıştırın!

NOT: E5 modeli query ve passage için prefix kullanır!
- Document: "passage: ..."
- Query: "query: ..."
"""

# ==============================================================================
# HÜCRE 1: Kütüphaneleri Kur
# ==============================================================================
# !pip install -q sentence-transformers qdrant-client tqdm

# ==============================================================================
# HÜCRE 2: Google Drive'ı Bağla
# ==============================================================================
"""
from google.colab import drive
drive.mount('/content/drive')
"""

# ==============================================================================
# HÜCRE 3: Ayarlar
# ==============================================================================

# ⚠️ BU YOLU KENDİ DRIVE YAPINIZA GÖRE DÜZENLEYİN!
DATA_FILE = "/content/drive/MyDrive/Dersler - Projeler/Derin Öğrenme Dersi/3- e5-large Qdrant WholeDocument/temiz.jsonl"

# Alternatif: Dosyayı Colab'a yüklediyseniz
# DATA_FILE = "/content/temiz.jsonl"

# Model ayarları - E5-Large
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 64  # L4 GPU için optimize

# E5 modeli prefix'leri
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Qdrant ayarları  
QDRANT_PATH = "/content/qdrant_data"
COLLECTION_NAME = "recipes"

# ==============================================================================
# HÜCRE 4: Import ve GPU Kontrolü
# ==============================================================================

import os
import json
import time
import shutil
from typing import Dict, Any, Generator

import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

print("=" * 60)
print("🍳 RAG TARİF ARAMA SİSTEMİ - COLAB INDEXER (E5-Large)")
print("=" * 60)

print(f"\n📊 PyTorch version: {torch.__version__}")
print(f"📊 CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ GPU bulunamadı! Runtime > Change runtime type > L4 GPU seçin.")

# ==============================================================================
# HÜCRE 5: Yardımcı Fonksiyonlar
# ==============================================================================

def create_recipe_text(recipe: Dict[str, Any], add_prefix: bool = True) -> str:
    """
    Tarif verisinden embedding için metin oluştur
    E5 modeli için passage prefix eklenir
    """
    title = recipe.get("title", "")
    ingredients = recipe.get("ingredients", [])
    instructions = recipe.get("instructions", [])
    
    ingredients_text = ", ".join(ingredients)
    instructions_text = " ".join(instructions)
    
    text = f"""Tarif: {title}

Malzemeler: {ingredients_text}

Yapılışı: {instructions_text}"""
    
    # E5 modeli için passage prefix ekle
    if add_prefix:
        text = f"{PASSAGE_PREFIX}{text}"
    
    return text


def load_recipes(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """JSONL dosyasından tarifleri yükle"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_recipes(file_path: str) -> int:
    """Toplam tarif sayısını hesapla"""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def batch_iterator(items, batch_size: int):
    """Generator'ı batch'lere böl"""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# ==============================================================================
# HÜCRE 6: Veri Dosyasını Kontrol Et
# ==============================================================================

print("\n" + "=" * 60)
print("📁 VERİ DOSYASI KONTROLÜ")
print("=" * 60)

if not os.path.exists(DATA_FILE):
    print(f"❌ Dosya bulunamadı: {DATA_FILE}")
    print("\n💡 Çözüm önerileri:")
    print("1. Google Drive'ı bağladığınızdan emin olun")
    print("2. DATA_FILE yolunu kontrol edin")
    print("3. Dosyayı manuel olarak Colab'a yükleyin")
    raise FileNotFoundError(f"Dosya bulunamadı: {DATA_FILE}")

file_size = os.path.getsize(DATA_FILE) / (1024 * 1024)
total_recipes = count_recipes(DATA_FILE)

print(f"✅ Dosya bulundu: {DATA_FILE}")
print(f"📊 Dosya boyutu: {file_size:.2f} MB")
print(f"📊 Toplam tarif sayısı: {total_recipes:,}")

# Örnek tarif göster
sample = next(load_recipes(DATA_FILE))
print(f"\n📋 Örnek tarif:")
print(f"   Başlık: {sample['title']}")
print(f"   Malzeme sayısı: {len(sample['ingredients'])}")
print(f"   Adım sayısı: {len(sample['instructions'])}")

# ==============================================================================
# HÜCRE 7: Modeli Yükle
# ==============================================================================

print("\n" + "=" * 60)
print("🤖 MODEL YÜKLEME (E5-Large)")
print("=" * 60)

print(f"🔄 Model yükleniyor: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

if torch.cuda.is_available():
    model = model.to('cuda')
    print("✅ Model GPU'ya yüklendi!")

EMBEDDING_DIM = model.get_sentence_embedding_dimension()
print(f"📊 Embedding boyutu: {EMBEDDING_DIM}")

# ==============================================================================
# HÜCRE 8: Qdrant Veritabanını Oluştur
# ==============================================================================

print("\n" + "=" * 60)
print("🗄️ QDRANT VERİTABANI")
print("=" * 60)

# Qdrant client oluştur
client = QdrantClient(path=QDRANT_PATH)

# Mevcut collection varsa sil
collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in collections:
    print(f"🗑️ Mevcut collection siliniyor: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

# Yeni collection oluştur
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_DIM,
        distance=Distance.COSINE
    )
)

print(f"✅ Collection oluşturuldu: {COLLECTION_NAME}")

# ==============================================================================
# HÜCRE 9: İNDEXLEME 🚀
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 TARİF İNDEXLEME BAŞLIYOR (E5-Large + Passage Prefix)")
print("=" * 60)

recipes_generator = load_recipes(DATA_FILE)

current_id = 0
total_indexed = 0
start_time = time.time()

with tqdm(total=total_recipes, desc="İndexleniyor", unit="tarif") as pbar:
    for batch in batch_iterator(recipes_generator, BATCH_SIZE):
        # Metinleri oluştur (passage prefix ile)
        texts = [create_recipe_text(r, add_prefix=True) for r in batch]
        
        # Embedding oluştur (GPU'da)
        embeddings = model.encode(
            texts, 
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Qdrant point'leri oluştur
        points = []
        for i, (recipe, embedding) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=current_id + i,
                vector=embedding.tolist(),
                payload={
                    "title": recipe.get("title", ""),
                    "url": recipe.get("url", ""),
                    "ingredients": recipe.get("ingredients", []),
                    "instructions": recipe.get("instructions", []),
                    "ingredient_count": len(recipe.get("ingredients", [])),
                    "instruction_count": len(recipe.get("instructions", []))
                }
            )
            points.append(point)
        
        # Veritabanına ekle
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        current_id += len(batch)
        total_indexed += len(batch)
        pbar.update(len(batch))

elapsed_time = time.time() - start_time

print("\n" + "=" * 60)
print("✅ İNDEXLEME TAMAMLANDI!")
print("=" * 60)
print(f"📊 Toplam indexlenen tarif: {total_indexed:,}")
print(f"⏱️ Geçen süre: {elapsed_time:.2f} saniye")
print(f"⚡ Hız: {total_indexed / elapsed_time:.2f} tarif/saniye")

# ==============================================================================
# HÜCRE 10: Veritabanını Doğrula
# ==============================================================================

print("\n" + "=" * 60)
print("🔍 VERİTABANI DOĞRULAMA")
print("=" * 60)

info = client.get_collection(COLLECTION_NAME)

print(f"📊 Collection: {COLLECTION_NAME}")
print(f"📊 Vektör sayısı: {info.points_count:,}")
print(f"📊 Durum: {info.status}")

# Test araması - E5 için query prefix kullan!
print("\n🔍 Test araması: 'tavuklu makarna'")
query_with_prefix = f"{QUERY_PREFIX}tavuklu makarna"
query_vector = model.encode(query_with_prefix).tolist()

# Yeni Qdrant API - query_points kullan
response = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
)

print("\n📋 Sonuçlar:")
for i, result in enumerate(response.points, 1):
    print(f"\n[{i}] {result.payload['title']}")
    print(f"    Skor: {result.score:.4f}")
    print(f"    Malzemeler: {', '.join(result.payload['ingredients'][:3])}...")

# ==============================================================================
# HÜCRE 11: Veritabanını Zip'le
# ==============================================================================

print("\n" + "=" * 60)
print("📦 VERİTABANI PAKETLEME")
print("=" * 60)

output_zip = "/content/qdrant_data.zip"

print("📦 Veritabanı zip'leniyor...")
shutil.make_archive("/content/qdrant_data", 'zip', QDRANT_PATH)

zip_size = os.path.getsize(output_zip) / (1024 * 1024)
print(f"✅ Zip dosyası oluşturuldu: {output_zip}")
print(f"📊 Zip boyutu: {zip_size:.2f} MB")

# ==============================================================================
# HÜCRE 12: İndir veya Drive'a Kaydet
# ==============================================================================

print("\n" + "=" * 60)
print("📥 İNDİRME SEÇENEKLERİ")
print("=" * 60)

print("""
Seçenek 1 - Doğrudan İndir (aşağıdaki kodu çalıştırın):
---------------------------------------------------------
from google.colab import files
files.download('/content/qdrant_data.zip')


Seçenek 2 - Google Drive'a Kaydet:
---------------------------------------------------------
import shutil
DRIVE_PATH = "/content/drive/MyDrive/Dersler - Projeler/Derin Öğrenme Dersi/3- e5-large Qdrant WholeDocument/qdrant_data.zip"
shutil.copy('/content/qdrant_data.zip', DRIVE_PATH)
print(f"✅ Drive'a kaydedildi: {DRIVE_PATH}")
""")

print("\n" + "=" * 60)
print("🎉 İŞLEM TAMAMLANDI!")
print("=" * 60)
print("""
Sonraki Adımlar:
1. qdrant_data.zip dosyasını indirin
2. Proje klasörünüze çıkarın (extract)
3. Yerel bilgisayarınızda çalıştırın:

   cd "E:\\Drive'ım\\Dersler - Projeler\\Derin Öğrenme Dersi\\3- e5-large Qdrant WholeDocument"
   .\\venv\\Scripts\\Activate.ps1
   python main.py search
""")


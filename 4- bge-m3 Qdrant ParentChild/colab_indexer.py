# -*- coding: utf-8 -*-
"""
🍳 RAG Tarif Arama Sistemi - Colab Indexer (Parent-Child)
=========================================================
Bu script'i Google Colab'da L4 GPU ile çalıştırın.

Kullanım:
1. Colab'da yeni notebook açın
2. Runtime > Change runtime type > L4 GPU seçin
3. Her hücreyi sırayla çalıştırın
4. qdrant_data.zip dosyasını indirip projeye çıkarın
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
DATA_FILE = "/content/drive/MyDrive/Dersler - Projeler/Derin Öğrenme Dersi/4- bge-m3 Qdrant ParentChild/temiz.jsonl"

# Alternatif: Dosyayı Colab'a yüklediyseniz
# DATA_FILE = "/content/temiz.jsonl"

# Model ayarları
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 64  # L4 GPU için optimize
EMBEDDING_DIM = 1024

# Qdrant ayarları  
QDRANT_PATH = "/content/qdrant_data"
COLLECTION_NAME = "recipes_parent_child"

# Parent-Child Chunking ayarları
CHUNK_TYPE_INGREDIENTS = "ingredients"
CHUNK_TYPE_INSTRUCTIONS = "instructions"
CHUNKS_PER_RECIPE = 2

# ==============================================================================
# HÜCRE 4: Import ve GPU Kontrolü
# ==============================================================================

import os
import json
import time
import shutil
from typing import Dict, Any, Generator, List, Tuple

import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

print("=" * 60)
print("🍳 RAG TARİF ARAMA SİSTEMİ - PARENT-CHILD INDEXER")
print("=" * 60)

print(f"\n📊 PyTorch version: {torch.__version__}")
print(f"📊 CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ GPU bulunamadı! Runtime > Change runtime type > L4 GPU seçin.")

# ==============================================================================
# HÜCRE 5: Chunk Oluşturma Fonksiyonları
# ==============================================================================

def create_ingredients_chunk(recipe: Dict[str, Any]) -> str:
    """Malzeme chunk'ı: Başlık + Malzemeler"""
    title = recipe.get("title", "")
    ingredients = recipe.get("ingredients", [])
    ingredients_text = ", ".join(ingredients)
    return f"""Tarif: {title}

Malzemeler: {ingredients_text}"""


def create_instructions_chunk(recipe: Dict[str, Any]) -> str:
    """Talimat chunk'ı: Başlık + Yapılış"""
    title = recipe.get("title", "")
    instructions = recipe.get("instructions", [])
    instructions_text = " ".join(instructions)
    return f"""Tarif: {title}

Yapılışı: {instructions_text}"""


def create_chunks(recipe: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Tarif için chunk'ları oluştur: [(chunk_type, text), ...]"""
    return [
        (CHUNK_TYPE_INGREDIENTS, create_ingredients_chunk(recipe)),
        (CHUNK_TYPE_INSTRUCTIONS, create_instructions_chunk(recipe))
    ]


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
total_chunks = total_recipes * CHUNKS_PER_RECIPE

print(f"✅ Dosya bulundu: {DATA_FILE}")
print(f"📊 Dosya boyutu: {file_size:.2f} MB")
print(f"📊 Toplam tarif sayısı: {total_recipes:,}")
print(f"📊 Oluşturulacak chunk sayısı: {total_chunks:,}")

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
print("🤖 MODEL YÜKLEME")
print("=" * 60)

print(f"🔄 Model yükleniyor: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

if torch.cuda.is_available():
    model = model.to('cuda')
    print("✅ Model GPU'ya yüklendi!")

print(f"📊 Embedding boyutu: {model.get_sentence_embedding_dimension()}")

# ==============================================================================
# HÜCRE 8: Qdrant Veritabanını Oluştur
# ==============================================================================

print("\n" + "=" * 60)
print("🗄️ QDRANT VERİTABANI")
print("=" * 60)

# Mevcut klasörü temizle
if os.path.exists(QDRANT_PATH):
    shutil.rmtree(QDRANT_PATH)
    print(f"🗑️ Mevcut veritabanı silindi: {QDRANT_PATH}")

# Qdrant client oluştur
client = QdrantClient(path=QDRANT_PATH)

# Collection oluştur
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_DIM,
        distance=Distance.COSINE
    )
)

print(f"✅ Collection oluşturuldu: {COLLECTION_NAME}")

# ==============================================================================
# HÜCRE 9: PARENT-CHILD İNDEXLEME 🚀
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 PARENT-CHILD İNDEXLEME BAŞLIYOR")
print("=" * 60)

recipes_generator = load_recipes(DATA_FILE)

current_parent_id = 0
total_indexed_chunks = 0
start_time = time.time()

with tqdm(total=total_recipes, desc="İndexleniyor", unit="tarif") as pbar:
    for batch in batch_iterator(recipes_generator, BATCH_SIZE):
        # Tüm chunk metinlerini topla
        all_chunk_info = []  # [(recipe_idx, chunk_idx, chunk_type, text), ...]
        
        for recipe_idx, recipe in enumerate(batch):
            chunks = create_chunks(recipe)
            for chunk_idx, (chunk_type, chunk_text) in enumerate(chunks):
                all_chunk_info.append((recipe_idx, chunk_idx, chunk_type, chunk_text))
        
        # Chunk metinlerini çıkar
        chunk_texts = [c[3] for c in all_chunk_info]
        
        # Toplu embedding (GPU'da)
        embeddings = model.encode(
            chunk_texts, 
            batch_size=BATCH_SIZE * 2,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Qdrant point'leri oluştur
        points = []
        
        for (recipe_idx, chunk_idx, chunk_type, _), embedding in zip(all_chunk_info, embeddings):
            recipe = batch[recipe_idx]
            parent_id = current_parent_id + recipe_idx
            point_id = parent_id * CHUNKS_PER_RECIPE + chunk_idx
            
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    # Parent bilgileri
                    "parent_id": parent_id,
                    "title": recipe.get("title", ""),
                    "url": recipe.get("url", ""),
                    "ingredients": recipe.get("ingredients", []),
                    "instructions": recipe.get("instructions", []),
                    
                    # Chunk bilgileri
                    "chunk_type": chunk_type,
                    "chunk_idx": chunk_idx,
                    
                    # Ek alanlar
                    "ingredient_count": len(recipe.get("ingredients", [])),
                    "instruction_count": len(recipe.get("instructions", []))
                }
            )
            points.append(point)
        
        # Qdrant'a ekle
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        current_parent_id += len(batch)
        total_indexed_chunks += len(points)
        pbar.update(len(batch))

elapsed_time = time.time() - start_time

print("\n" + "=" * 60)
print("✅ PARENT-CHILD İNDEXLEME TAMAMLANDI!")
print("=" * 60)
print(f"📊 Toplam indexlenen tarif: {current_parent_id:,}")
print(f"📊 Toplam indexlenen chunk: {total_indexed_chunks:,}")
print(f"⏱️ Geçen süre: {elapsed_time:.2f} saniye")
print(f"⚡ Hız: {current_parent_id / elapsed_time:.2f} tarif/saniye")

# ==============================================================================
# HÜCRE 10: Veritabanını Doğrula
# ==============================================================================

print("\n" + "=" * 60)
print("🔍 VERİTABANI DOĞRULAMA")
print("=" * 60)

info = client.get_collection(COLLECTION_NAME)
print(f"📊 Collection: {COLLECTION_NAME}")
print(f"📊 Chunk sayısı: {info.points_count:,}")
print(f"📊 Tarif sayısı: {info.points_count // CHUNKS_PER_RECIPE:,}")

# Test araması
print("\n🔍 Test araması: 'tavuklu makarna'")
query_vector = model.encode("tavuklu makarna").tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5
)

# Parent'a göre grupla
parent_results = {}
for r in results.points:
    parent_id = r.payload.get("parent_id")
    if parent_id not in parent_results or r.score > parent_results[parent_id]["score"]:
        parent_results[parent_id] = {
            "title": r.payload.get("title"),
            "score": r.score,
            "chunk_type": r.payload.get("chunk_type")
        }

print("\n📋 Sonuçlar (Parent bazlı):")
for i, (pid, data) in enumerate(sorted(parent_results.items(), key=lambda x: x[1]["score"], reverse=True)[:3], 1):
    print(f"\n[{i}] {data['title']}")
    print(f"    Skor: {data['score']:.4f}")
    print(f"    Eşleşen chunk: {data['chunk_type']}")

# ==============================================================================
# HÜCRE 11: Veritabanını Zip'le
# ==============================================================================

print("\n" + "=" * 60)
print("📦 VERİTABANI PAKETLEME")
print("=" * 60)

# Client'ı kapat
client.close()

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
DRIVE_PATH = "/content/drive/MyDrive/Dersler - Projeler/Derin Öğrenme Dersi/4- bge-m3 Qdrant ParentChild/qdrant_data.zip"
shutil.copy('/content/qdrant_data.zip', DRIVE_PATH)
print(f"✅ Drive'a kaydedildi: {DRIVE_PATH}")
""")

print("\n" + "=" * 60)
print("🎉 İŞLEM TAMAMLANDI!")
print("=" * 60)
print("""
Sonraki Adımlar:
1. qdrant_data.zip dosyasını indirin
2. Proje klasörüne çıkarın (extract)
3. Yerel bilgisayarınızda çalıştırın:

   cd "E:\\Drive'ım\\Dersler - Projeler\\Derin Öğrenme Dersi\\4- bge-m3 Qdrant ParentChild"
   .\\venv\\Scripts\\Activate.ps1
   python main.py search
   
Parent-Child Arama Komutları:
   /malzeme tavuk, patates   → Malzeme chunk'larında ara
   /yontem fırında          → Talimat chunk'larında ara
""")


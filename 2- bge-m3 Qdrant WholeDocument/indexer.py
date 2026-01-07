"""
Tarif Indexleme Modülü
======================
JSONL dosyasını okuyup Qdrant'a indexleme
"""

import json
from typing import Generator, Dict, Any, List
from tqdm import tqdm
from config import DATA_FILE, BATCH_SIZE, INDEX_BATCH_SIZE
from embedder import get_embedder
from database import get_database


def load_recipes(file_path: str = None) -> Generator[Dict[str, Any], None, None]:
    """
    JSONL dosyasından tarifleri yükle (generator)
    
    Args:
        file_path: JSONL dosya yolu (varsayılan: config'den)
    
    Yields:
        Her satırdaki tarif dictionary
    """
    path = file_path or DATA_FILE
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_recipes(file_path: str = None) -> int:
    """Toplam tarif sayısını hesapla"""
    path = file_path or DATA_FILE
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def batch_iterator(items: Generator, batch_size: int) -> Generator[List, None, None]:
    """Generator'ı batch'lere böl"""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def index_all_recipes(recreate: bool = True, file_path: str = None):
    """
    Tüm tarifleri indexle
    
    Args:
        recreate: True ise mevcut collection silinip yeniden oluşturulur
        file_path: JSONL dosya yolu
    """
    print("=" * 60)
    print("🚀 TARİF İNDEXLEME BAŞLIYOR")
    print("=" * 60)
    
    # Toplam tarif sayısını hesapla
    print("\n📊 Tarif sayısı hesaplanıyor...")
    total_recipes = count_recipes(file_path)
    print(f"📊 Toplam tarif sayısı: {total_recipes:,}")
    
    # Embedder ve Database başlat
    embedder = get_embedder()
    db = get_database()
    
    # Collection oluştur
    db.create_collection(recreate=recreate)
    
    # Tarifleri batch'ler halinde işle
    print(f"\n📥 Tarifler işleniyor (batch boyutu: {BATCH_SIZE})...")
    
    recipes_generator = load_recipes(file_path)
    current_id = 0
    total_indexed = 0
    
    # Progress bar
    with tqdm(total=total_recipes, desc="İndexleniyor", unit="tarif") as pbar:
        for batch in batch_iterator(recipes_generator, BATCH_SIZE):
            # Embedding oluştur
            vectors = embedder.embed_recipes(batch)
            
            # Veritabanına ekle
            inserted = db.insert_recipes(batch, vectors, start_id=current_id)
            
            current_id += len(batch)
            total_indexed += inserted
            pbar.update(len(batch))
    
    print("\n" + "=" * 60)
    print("✅ İNDEXLEME TAMAMLANDI!")
    print("=" * 60)
    print(f"📊 Toplam indexlenen tarif: {total_indexed:,}")
    
    # Collection bilgisi
    info = db.get_collection_info()
    print(f"📊 Veritabanı vektör sayısı: {info.get('points_count', 'N/A'):,}")
    
    return total_indexed


def verify_index():
    """Index'in doğru çalıştığını kontrol et"""
    print("\n🔍 Index doğrulaması yapılıyor...")
    
    db = get_database()
    info = db.get_collection_info()
    
    if not info.get("exists"):
        print("❌ Collection bulunamadı!")
        return False
    
    print(f"✅ Collection mevcut")
    print(f"📊 Vektör sayısı: {info.get('points_count', 0):,}")
    print(f"📊 Durum: {info.get('status', 'N/A')}")
    
    # Örnek bir kayıt getir
    sample = db.get_recipe_by_id(0)
    if sample:
        print(f"\n📋 Örnek tarif (ID: 0):")
        print(f"   Başlık: {sample['title']}")
        print(f"   Malzeme sayısı: {len(sample['ingredients'])}")
        print(f"   Adım sayısı: {len(sample['instructions'])}")
        return True
    
    return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_index()
    else:
        # Full indexleme
        index_all_recipes(recreate=True)
        verify_index()


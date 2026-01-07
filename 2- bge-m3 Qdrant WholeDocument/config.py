"""
Proje Konfigürasyonu
====================
RAG Tarif Arama Sistemi için tüm ayarlar
"""

from pathlib import Path

# ============================================================
# DOSYA YOLLARI
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "temiz.jsonl"
QDRANT_PATH = BASE_DIR / "qdrant_data"

# ============================================================
# BGE-M3 MODEL AYARLARI
# ============================================================
MODEL_NAME = "BAAI/bge-m3"
USE_FP16 = True  # GPU bellek optimizasyonu için
EMBEDDING_DIM = 1024  # BGE-M3 dense vector boyutu

# ============================================================
# QDRANT AYARLARI
# ============================================================
COLLECTION_NAME = "recipes"
DISTANCE_METRIC = "Cosine"  # Cosine, Euclid, Dot

# ============================================================
# INDEXLEME AYARLARI
# ============================================================
BATCH_SIZE = 32  # Embedding batch boyutu
INDEX_BATCH_SIZE = 100  # Qdrant'a yazma batch boyutu

# ============================================================
# ARAMA AYARLARI
# ============================================================
DEFAULT_TOP_K = 5  # Varsayılan sonuç sayısı
SCORE_THRESHOLD = 0.3  # Minimum benzerlik skoru (0-1 arası)


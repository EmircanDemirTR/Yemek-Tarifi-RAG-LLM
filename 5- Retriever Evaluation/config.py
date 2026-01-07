"""
Retriever Evaluation Konfigürasyonu
===================================
Tüm retriever sistemlerini değerlendirmek için ayarlar
"""

from pathlib import Path

# ============================================================
# DOSYA YOLLARI
# ============================================================
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
EVALUATION_SET_PATH = BASE_DIR / "evaluation_set.json"
RESULTS_DIR = BASE_DIR / "results"

# ============================================================
# DEĞERLENDİRİLECEK SİSTEMLER
# ============================================================
RETRIEVER_SYSTEMS = {
    "bge_m3_wholedoc": {
        "name": "BGE-M3 WholeDocument",
        "path": PROJECT_DIR / "2- bge-m3 Qdrant WholeDocument",
        "embedding_model": "BAAI/bge-m3",
        "chunking": "WholeDocument"
    },
    "e5_large_wholedoc": {
        "name": "E5-Large WholeDocument", 
        "path": PROJECT_DIR / "3- e5-large Qdrant WholeDocument",
        "embedding_model": "intfloat/multilingual-e5-large",
        "chunking": "WholeDocument"
    },
    "bge_m3_parentchild": {
        "name": "BGE-M3 Parent-Child",
        "path": PROJECT_DIR / "4- bge-m3 Qdrant ParentChild",
        "embedding_model": "BAAI/bge-m3",
        "chunking": "Parent-Child"
    }
}

# ============================================================
# DEĞERLENDİRME AYARLARI
# ============================================================
K_VALUES = [1, 3, 5, 10]  # Test edilecek top-k değerleri
DEFAULT_K = 5

# Metrik eşik değerleri
SIMILARITY_THRESHOLD = 0.3

# ============================================================
# ÇIKTI AYARLARI
# ============================================================
SAVE_DETAILED_RESULTS = True
GENERATE_CHARTS = True


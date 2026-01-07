"""
RAG Pipeline Konfigürasyonu
"""
import os
from pathlib import Path

# .env dosyasından API key'leri yükle
try:
    from dotenv import load_dotenv
    # Proje kök dizinindeki .env dosyasını yükle (override=True cache'i geçersiz kılar)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, override=True)
except ImportError:
    pass  # python-dotenv yüklü değilse devam et

# Dosya yolları
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_FILE = PROJECT_DIR / "1- Veri Kazıma ve Temizleme" / "temiz.jsonl"

# Retriever ayarları - BGE-M3 WholeDocument kullanacağız (en iyi MRR)
RETRIEVER_PATH = PROJECT_DIR / "2- bge-m3 Qdrant WholeDocument"
DEFAULT_TOP_K = 5

# OpenAI API ayarları
# API key'i environment variable olarak ayarlayın: set OPENAI_API_KEY=your-key-here
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# Groq API ayarları (yedek)
# API key'i environment variable olarak ayarlayın: set GROQ_API_KEY=your-key-here
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Ollama ayarları
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODELS = {
    "qwen2:1.5b": {"name": "Qwen2 1.5B", "size": "1.5B", "speed": "fast"},
    "llama3.2:3b": {"name": "Llama 3.2 3B", "size": "3B", "speed": "medium"},
    "phi3:mini": {"name": "Phi-3 Mini", "size": "3.8B", "speed": "medium"},
    "gemma2:2b": {"name": "Gemma2 2B", "size": "2B", "speed": "fast"},
}

# Varsayılan model
DEFAULT_LOCAL_MODEL = "qwen2:1.5b"

# Prompt ayarları
MAX_CONTEXT_LENGTH = 2000  # Karakter
MAX_RESPONSE_TOKENS = 500

# Arama ayarları (Retriever için)
SCORE_THRESHOLD = 0.3


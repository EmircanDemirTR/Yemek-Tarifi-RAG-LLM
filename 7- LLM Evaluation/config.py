"""
LLM Evaluation Konfigürasyonu
"""
from pathlib import Path

# Dosya yolları
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
EVALUATION_SET_PATH = PROJECT_DIR / "5- Retriever Evaluation" / "evaluation_set.json"
RESULTS_DIR = BASE_DIR / "results"

# Değerlendirilecek modeller
MODELS_TO_EVALUATE = {
    "groq": {
        "name": "Groq Llama 3.3 70B",
        "provider": "groq",
        "model": None  # Varsayılan kullanılır
    },
    "qwen2": {
        "name": "Ollama Qwen2 1.5B",
        "provider": "ollama", 
        "model": "qwen2:1.5b"
    },
    "llama3.2": {
        "name": "Ollama Llama 3.2 3B",
        "provider": "ollama",
        "model": "llama3.2:3b"
    },
    "phi3": {
        "name": "Ollama Phi-3 Mini",
        "provider": "ollama",
        "model": "phi3:mini"
    },
    "gemma2": {
        "name": "Ollama Gemma2 2B",
        "provider": "ollama",
        "model": "gemma2:2b"
    }
}

# Hızlı test için sadece bu modeller
QUICK_TEST_MODELS = ["groq", "qwen2"]

# Değerlendirme ayarları
MAX_QUESTIONS = 10  # Hızlı test için
FULL_QUESTIONS = 50  # Tam test için (impossible hariç)


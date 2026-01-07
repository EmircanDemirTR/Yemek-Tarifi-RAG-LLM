"""
Ollama Lokal LLM Entegrasyonu
"""
import os
import sys
import time
import requests
import json

# Windows terminal encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

from config import OLLAMA_HOST, OLLAMA_MODELS, DEFAULT_LOCAL_MODEL


class OllamaLLM:
    def __init__(self, model: str = None, host: str = None):
        self.host = host or OLLAMA_HOST
        self.model = model or DEFAULT_LOCAL_MODEL
        self.api_url = f"{self.host}/api/generate"
    
    def generate(self, prompt: str, max_tokens: int = 500) -> dict:
        """
        Ollama ile cevap üret
        
        Returns:
            {
                "response": str,
                "latency_ms": float,
                "tokens": int,
                "model": str,
                "tokens_per_second": float
            }
        """
        start_time = time.time()
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.3,
                    }
                },
                timeout=120  # 2 dakika timeout
            )
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 1)  # nanoseconds
                
                # Token/s hesapla
                tokens_per_second = (eval_count / (eval_duration / 1e9)) if eval_duration > 0 else 0
                
                return {
                    "response": response_text,
                    "latency_ms": latency,
                    "tokens": eval_count,
                    "model": self.model,
                    "provider": "ollama",
                    "tokens_per_second": tokens_per_second
                }
            else:
                return {
                    "response": f"HTTP Hata: {response.status_code}",
                    "latency_ms": latency,
                    "tokens": 0,
                    "model": self.model,
                    "provider": "ollama",
                    "error": f"HTTP {response.status_code}"
                }
        
        except requests.exceptions.ConnectionError:
            return {
                "response": "Ollama bağlantı hatası! 'ollama serve' çalıştırın.",
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens": 0,
                "model": self.model,
                "provider": "ollama",
                "error": "connection_error"
            }
        except Exception as e:
            return {
                "response": f"Hata: {str(e)}",
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens": 0,
                "model": self.model,
                "provider": "ollama",
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """Ollama bağlantısını test et"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """Yüklü modelleri listele"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except:
            return []
    
    def is_model_available(self, model: str = None) -> bool:
        """Model yüklü mü kontrol et"""
        model = model or self.model
        models = self.list_models()
        return any(model in m for m in models)


def get_available_models() -> dict:
    """Kullanılabilir Ollama modellerini getir"""
    llm = OllamaLLM()
    installed = llm.list_models()
    
    available = {}
    for model_id, info in OLLAMA_MODELS.items():
        is_installed = any(model_id.split(":")[0] in m for m in installed)
        available[model_id] = {
            **info,
            "installed": is_installed
        }
    
    return available


if __name__ == "__main__":
    # Test
    print("Ollama LLM Test")
    print("-" * 40)
    
    llm = OllamaLLM()
    
    if llm.test_connection():
        print("✓ Ollama bağlantısı başarılı!")
        
        models = llm.list_models()
        print(f"\nYüklü modeller: {models}")
        
        if llm.is_model_available():
            print(f"\n{llm.model} ile test ediliyor...")
            result = llm.generate("Merhaba, sadece 'OK' yaz.")
            print(f"Cevap: {result['response'][:100]}")
            print(f"Latency: {result['latency_ms']:.0f}ms")
            if result.get('tokens_per_second'):
                print(f"Hız: {result['tokens_per_second']:.1f} token/s")
        else:
            print(f"\n✗ {llm.model} modeli yüklü değil!")
            print(f"  Yüklemek için: ollama pull {llm.model}")
    else:
        print("✗ Ollama bağlantısı başarısız!")
        print("  'ollama serve' komutunu çalıştırın.")


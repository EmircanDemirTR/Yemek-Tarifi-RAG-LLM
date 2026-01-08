"""
OpenAI API LLM Entegrasyonu
"""
import os
import sys
import time

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Windows terminal encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL


class OpenAILLM:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY gerekli!")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or OPENAI_MODEL
    
    def generate(self, prompt: str, max_tokens: int = 500) -> dict:
        """OpenAI API ile cevap üret"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Sen bir Türk mutfağı uzmanısın. Kısa ve öz cevaplar ver."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            
            latency = (time.time() - start_time) * 1000
            response_text = response.choices[0].message.content
            total_tokens = response.usage.total_tokens if response.usage else 0
            
            return {
                "response": response_text,
                "latency_ms": latency,
                "tokens": total_tokens,
                "model": self.model,
                "provider": "openai"
            }
        
        except Exception as e:
            return {
                "response": f"Hata: {str(e)}",
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens": 0,
                "model": self.model,
                "provider": "openai",
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """API bağlantısını test et"""
        try:
            result = self.generate("Merhaba, sadece 'OK' yaz.", max_tokens=10)
            return "error" not in result
        except:
            return False


if __name__ == "__main__":
    print("OpenAI API Test")
    print("-" * 40)
    
    try:
        llm = OpenAILLM()
        if llm.test_connection():
            print("✓ Bağlantı başarılı!")
            
            result = llm.generate("Mercimek çorbası nasıl yapılır? Kısa cevap ver.")
            print(f"\nCevap: {result['response'][:200]}...")
            print(f"Latency: {result['latency_ms']:.0f}ms")
            print(f"Tokens: {result['tokens']}")
        else:
            print("✗ Bağlantı başarısız!")
    except ValueError as e:
        print(f"✗ {e}")



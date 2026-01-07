"""
Groq API LLM Entegrasyonu
"""
import os
import sys
import time

# Windows terminal encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


class GroqLLM:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY gerekli! Environment variable olarak ayarlayın.")
        self.client = Groq(api_key=self.api_key)
        self.model = GROQ_MODEL
    
    def generate(self, prompt: str, max_tokens: int = 500, retry_count: int = 3) -> dict:
        """
        Groq API ile cevap üret (rate limit için retry destekli)
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(retry_count):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                
                latency = (time.time() - start_time) * 1000
                response_text = completion.choices[0].message.content
                total_tokens = completion.usage.total_tokens if completion.usage else 0
                
                return {
                    "response": response_text,
                    "latency_ms": latency,
                    "tokens": total_tokens,
                    "model": self.model,
                    "provider": "groq"
                }
            
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Rate limit hatası ise bekle ve tekrar dene
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = (attempt + 1) * 10  # 10, 20, 30 saniye
                    print(f" [Rate limit, {wait_time}s bekleniyor...]", end="", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    break
        
        return {
            "response": f"Hata: {str(last_error)}",
            "latency_ms": (time.time() - start_time) * 1000,
            "tokens": 0,
            "model": self.model,
            "provider": "groq",
            "error": str(last_error)
        }
    
    def test_connection(self) -> bool:
        """API bağlantısını test et"""
        try:
            result = self.generate("Merhaba, sadece 'OK' yaz.", max_tokens=10)
            return "error" not in result
        except:
            return False


if __name__ == "__main__":
    # Test
    print("Groq API Test")
    print("-" * 40)
    
    try:
        llm = GroqLLM()
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


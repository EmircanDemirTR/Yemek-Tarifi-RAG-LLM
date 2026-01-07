"""
RAG Pipeline - Retriever + LLM
3 mod destekler: Retriever-Only, LLM-Only, RAG (Retriever + LLM)
"""
import os
import sys

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

# Windows terminal encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Optional, List, Dict

from config import RETRIEVER_PATH, DEFAULT_TOP_K
from prompt_templates import create_rag_prompt, create_llm_only_prompt

# Global retriever instance (Qdrant lock sorunu için)
_global_retriever = None


def get_global_retriever():
    """Global retriever instance döndür"""
    global _global_retriever
    
    if _global_retriever is None:
        original_path = sys.path.copy()
        
        try:
            retriever_path = str(RETRIEVER_PATH)
            sys.path.insert(0, retriever_path)
            
            if 'config' in sys.modules:
                del sys.modules['config']
            
            from searcher import RecipeSearcher
            _global_retriever = RecipeSearcher()
            
        finally:
            sys.path = original_path
            if 'config' in sys.modules:
                del sys.modules['config']
            import config
    
    return _global_retriever


class RAGPipeline:
    """
    RAG Pipeline - 3 mod destekler:
    1. retriever_only: Sadece retriever (LLM yok)
    2. llm_only: Sadece LLM (retriever yok)
    3. rag: Retriever + LLM
    """
    
    def __init__(self, llm_provider: str = "openai", llm_model: str = None):
        self.llm = None
        self.llm_provider = llm_provider
        self._init_llm(llm_provider, llm_model)
    
    def _init_llm(self, provider: str, model: str = None):
        """LLM'i başlat"""
        if provider == "openai":
            from llm_openai import OpenAILLM
            self.llm = OpenAILLM(model=model)
        elif provider == "groq":
            from llm_api import GroqLLM
            self.llm = GroqLLM()
        elif provider == "ollama":
            from llm_local import OllamaLLM
            self.llm = OllamaLLM(model=model)
        else:
            raise ValueError(f"Bilinmeyen provider: {provider}")
        self.llm_provider = provider
    
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Retriever ile arama yap (global instance kullan)"""
        retriever = get_global_retriever()
        return retriever.search(query, top_k=top_k)
    
    def query_retriever_only(self, question: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """Retriever-Only modu"""
        import time
        start = time.time()
        
        recipes = self.search(question, top_k=top_k)
        latency = (time.time() - start) * 1000
        
        return {
            "question": question,
            "retrieved_recipes": recipes,
            "latency_ms": latency,
            "mode": "retriever_only"
        }
    
    def query_llm_only(self, question: str) -> Dict:
        """LLM-Only modu: Sadece LLM, context yok"""
        prompt = create_llm_only_prompt(question)
        llm_result = self.llm.generate(prompt)
        
        return {
            "question": question,
            "answer": llm_result["response"],
            "llm_result": llm_result,
            "mode": "llm_only"
        }
    
    def query_rag(self, question: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """RAG modu: Retriever + LLM"""
        # 1. Retriever
        recipes = self.search(question, top_k=top_k)
        
        # 2. Prompt oluştur
        prompt = create_rag_prompt(question, recipes)
        
        # 3. LLM
        llm_result = self.llm.generate(prompt)
        
        # Context oluştur (metrik hesaplama için)
        context = " ".join([
            r.get("title", "") + " " + " ".join(r.get("ingredients", []))
            for r in recipes
        ])
        
        return {
            "question": question,
            "answer": llm_result["response"],
            "retrieved_recipes": recipes,
            "context": context,
            "llm_result": llm_result,
            "mode": "rag"
        }
    
    def switch_llm(self, provider: str, model: str = None):
        """LLM'i değiştir"""
        self._init_llm(provider, model)
    
    def get_llm_info(self) -> Dict:
        """Mevcut LLM bilgisi"""
        return {
            "provider": self.llm_provider,
            "model": getattr(self.llm, 'model', 'unknown')
        }
    
    def close(self):
        """Sadece LLM referansını temizle (retriever global)"""
        self.llm = None


if __name__ == "__main__":
    print("RAG Pipeline Test")
    print("=" * 60)
    
    rag = None
    try:
        print("\n1. OpenAI API Test (LLM-Only):")
        rag = RAGPipeline(llm_provider="openai")
        result = rag.query_llm_only("Mercimek çorbası nasıl yapılır?")
        print(f"   Cevap: {result['answer'][:150]}...")
        print(f"   Latency: {result['llm_result']['latency_ms']:.0f}ms")
        
        print("\n2. RAG Test (Retriever + LLM):")
        result = rag.query_rag("Karnıyarık tarifi")
        print(f"   Bulunan: {len(result['retrieved_recipes'])} tarif")
        print(f"   Cevap: {result['answer'][:150]}...")
        
        print("\n3. Retriever-Only Test:")
        result = rag.query_retriever_only("Baklava nasıl yapılır?")
        print(f"   Bulunan: {len(result['retrieved_recipes'])} tarif")
        for r in result['retrieved_recipes'][:3]:
            print(f"   - {r['title']} (skor: {r.get('score', 0):.3f})")
        
    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            rag.close()

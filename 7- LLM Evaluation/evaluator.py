"""
LLM Performans Değerlendirme
PDF Rehberine göre 3 seviye:
1. Retriever-Only (Recall@k, Hit@k, MRR@k) - zaten 5- klasöründe yapıldı
2. LLM-Only (EM, F1, Hallucination)
3. RAG + LLM (EM, F1, Hallucination)
"""
import os
import sys

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import json
import time
from pathlib import Path
from datetime import datetime

# Windows terminal encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

# Path ayarları
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
EVALUATION_SET_PATH = PROJECT_DIR / "5- Retriever Evaluation" / "evaluation_set.json"
RESULTS_DIR = BASE_DIR / "results"

# RAG Pipeline import
sys.path.insert(0, str(PROJECT_DIR / "6- RAG Pipeline"))
from rag_pipeline import RAGPipeline, get_global_retriever
from metrics import calculate_llm_metrics, aggregate_llm_metrics

# Retriever'ı baştan yükle (Qdrant lock sorunu için)
_retriever_initialized = False
def init_retriever_once():
    global _retriever_initialized
    if not _retriever_initialized:
        print("\n🔄 Retriever başlatılıyor (bir kez)...")
        try:
            get_global_retriever()
            _retriever_initialized = True
            print("✅ Retriever hazır!\n")
        except Exception as e:
            print(f"⚠️ Retriever başlatılamadı: {e}\n")

# Verbose mod - cevapları detaylı göster (default: açık)
VERBOSE_MODE = True

# Modeller - PDF'e göre: 1 API + 4 lokal (toplam 5)
MODELS = {
    "openai": {"name": "OpenAI GPT-4o-mini", "provider": "openai", "model": None},
    "qwen2": {"name": "Ollama Qwen2 1.5B", "provider": "ollama", "model": "qwen2:1.5b"},
    "llama3.2": {"name": "Ollama Llama 3.2 3B", "provider": "ollama", "model": "llama3.2:3b"},
    "phi3": {"name": "Ollama Phi-3 Mini", "provider": "ollama", "model": "phi3:mini"},
    "mistral": {"name": "Ollama Mistral 7B", "provider": "ollama", "model": "mistral:7b-instruct-q4_0"}
}


def load_questions(max_q: int = None):
    """Soruları yükle (impossible hariç)"""
    with open(EVALUATION_SET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = [q for q in data["questions"] if q.get("category") != "impossible"]
    return questions[:max_q] if max_q else questions


def evaluate_llm_only(model_key: str, questions: list) -> dict:
    """LLM-Only değerlendirme"""
    config = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"📊 {config['name']} - LLM-Only")
    print(f"{'='*60}")
    
    rag = None
    try:
        rag = RAGPipeline(llm_provider=config["provider"], llm_model=config.get("model"))
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return {"error": str(e)}
    
    results = []
    total_latency = 0
    
    try:
        for i, q in enumerate(questions, 1):
            question = q["question"]
            gold = q.get("gold_answer", "")
            keywords = q.get("keywords", [])
            expected = q.get("expected_recipes", [])
            
            print(f"  [{i}/{len(questions)}] {question[:40]}...", end=" ", flush=True)
            
            try:
                result = rag.query_llm_only(question)
                prediction = result["answer"]
                latency = result["llm_result"]["latency_ms"]
                total_latency += latency
                
                metrics = calculate_llm_metrics(
                    prediction, gold, 
                    keywords=keywords, 
                    expected_recipes=expected
                )
                metrics["latency_ms"] = latency
                metrics["question"] = question
                metrics["prediction"] = prediction[:300]
                results.append(metrics)
                
                score = metrics["combined_score"]
                status = "✓" if score > 0.2 else "○"
                hall = "H" if metrics["is_hallucination"] else ""
                print(f"{status} Score={score:.2f} {hall} | {latency:.0f}ms")
                
                # Verbose: Cevabı göster
                if VERBOSE_MODE:
                    print(f"      💬 Cevap: {prediction[:150]}{'...' if len(prediction) > 150 else ''}")
                
            except Exception as e:
                print(f"❌ {str(e)[:25]}")
                results.append({
                    "question": question, "error": str(e), 
                    "f1": 0, "exact_match": 0, "combined_score": 0,
                    "is_hallucination": True, "latency_ms": 0
                })
    finally:
        if rag:
            rag.close()
    
    agg = aggregate_llm_metrics([r for r in results if "error" not in r])
    agg["avg_latency_ms"] = total_latency / len(questions) if questions else 0
    
    print(f"\n📈 LLM-Only Sonuçları:")
    print(f"   F1: {agg.get('f1', 0):.2%} | Keyword: {agg.get('keyword_score', 0):.2%}")
    print(f"   Combined: {agg.get('combined_score', 0):.2%} | Hall: {agg.get('hallucination_rate', 0):.2%}")
    
    return {"model": config["name"], "mode": "llm_only", "aggregated": agg, "results": results}


def evaluate_rag(model_key: str, questions: list) -> dict:
    """RAG + LLM değerlendirme"""
    config = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"📊 {config['name']} - RAG + LLM")
    print(f"{'='*60}")
    
    rag = None
    try:
        rag = RAGPipeline(llm_provider=config["provider"], llm_model=config.get("model"))
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return {"error": str(e)}
    
    results = []
    total_latency = 0
    
    try:
        for i, q in enumerate(questions, 1):
            question = q["question"]
            gold = q.get("gold_answer", "")
            keywords = q.get("keywords", [])
            expected = q.get("expected_recipes", [])
            
            print(f"  [{i}/{len(questions)}] {question[:40]}...", end=" ", flush=True)
            
            try:
                result = rag.query_rag(question)
                prediction = result["answer"]
                context = result.get("context", "")
                latency = result["llm_result"]["latency_ms"]
                total_latency += latency
                
                metrics = calculate_llm_metrics(
                    prediction, gold,
                    context=context,
                    keywords=keywords,
                    expected_recipes=expected
                )
                metrics["latency_ms"] = latency
                metrics["question"] = question
                metrics["prediction"] = prediction[:300]
                metrics["num_recipes"] = len(result.get("retrieved_recipes", []))
                results.append(metrics)
                
                score = metrics["combined_score"]
                status = "✓" if score > 0.2 else "○"
                hall = "H" if metrics["is_hallucination"] else ""
                recipes_found = metrics["num_recipes"]
                print(f"{status} Score={score:.2f} {hall} | {recipes_found} tarif | {latency:.0f}ms")
                
                # Verbose: Cevabı ve bulunan tarifleri göster
                if VERBOSE_MODE:
                    print(f"      💬 Cevap: {prediction[:150]}{'...' if len(prediction) > 150 else ''}")
                    if recipes_found > 0:
                        recipe_titles = [r.get("title", "?")[:30] for r in result.get("retrieved_recipes", [])[:3]]
                        print(f"      📚 Tarifler: {', '.join(recipe_titles)}")
                
            except Exception as e:
                print(f"❌ {str(e)[:25]}")
                results.append({
                    "question": question, "error": str(e),
                    "f1": 0, "exact_match": 0, "combined_score": 0,
                    "is_hallucination": True, "latency_ms": 0
                })
    finally:
        if rag:
            rag.close()
    
    agg = aggregate_llm_metrics([r for r in results if "error" not in r])
    agg["avg_latency_ms"] = total_latency / len(questions) if questions else 0
    
    print(f"\n📈 RAG + LLM Sonuçları:")
    print(f"   F1: {agg.get('f1', 0):.2%} | Keyword: {agg.get('keyword_score', 0):.2%}")
    print(f"   Combined: {agg.get('combined_score', 0):.2%} | Hall: {agg.get('hallucination_rate', 0):.2%}")
    
    return {"model": config["name"], "mode": "rag", "aggregated": agg, "results": results}


def run_full_evaluation(model_keys: list = None, max_questions: int = 10):
    """Tam değerlendirme çalıştır"""
    print("\n" + "="*70)
    print("🍳 LLM Performans Değerlendirmesi")
    print("   PDF Rehberine göre: LLM-Only vs RAG + LLM")
    print("="*70)
    
    # Retriever'ı baştan yükle
    init_retriever_once()
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    if model_keys is None:
        model_keys = ["openai", "qwen2"]
    
    questions = load_questions(max_questions)
    print(f"\n📝 {len(questions)} soru | Modeller: {', '.join(model_keys)}")
    
    all_results = {}
    
    for key in model_keys:
        if key not in MODELS:
            continue
        
        # LLM-Only
        result = evaluate_llm_only(key, questions)
        all_results[f"{key}_llm_only"] = result
        
        # RAG + LLM
        result = evaluate_rag(key, questions)
        all_results[f"{key}_rag"] = result
    
    # Kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"evaluation_{timestamp}.json"
    
    save_data = {
        "timestamp": timestamp,
        "num_questions": len(questions),
        "models": model_keys,
        "results": {}
    }
    
    for k, v in all_results.items():
        if isinstance(v, dict) and "error" not in v:
            save_data["results"][k] = {
                "model": v.get("model"),
                "mode": v.get("mode"),
                "aggregated": v.get("aggregated", {})
            }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Sonuçlar: {results_file}")
    
    # Final karşılaştırma tablosu (PDF formatında)
    print("\n" + "="*100)
    print("📊 FINAL KARŞILAŞTIRMA TABLOSU (PDF Rehberi Formatı)")
    print("="*100)
    print(f"{'Model':<25} | {'Mode':<10} | {'EM':<7} | {'F1':<7} | {'Combined':<9} | {'Hall.':<7} | {'Latency':<10}")
    print("-"*100)
    
    for key, result in all_results.items():
        if "error" in result:
            continue
        agg = result.get("aggregated", {})
        print(f"{result.get('model', key)[:23]:<25} | {result.get('mode', ''):<10} | "
              f"{agg.get('exact_match', 0):.2%}  | {agg.get('f1', 0):.2%}  | "
              f"{agg.get('combined_score', 0):.2%}    | {agg.get('hallucination_rate', 0):.2%}  | "
              f"{agg.get('avg_latency_ms', 0):.0f}ms")
    
    print("="*100)
    
    # Karşılaştırma özeti
    print("\n📋 KARŞILAŞTIRMA ÖZETİ:")
    for key in model_keys:
        llm_only = all_results.get(f"{key}_llm_only", {}).get("aggregated", {})
        rag = all_results.get(f"{key}_rag", {}).get("aggregated", {})
        
        if llm_only and rag:
            llm_f1 = llm_only.get("combined_score", 0)
            rag_f1 = rag.get("combined_score", 0)
            improvement = ((rag_f1 - llm_f1) / llm_f1 * 100) if llm_f1 > 0 else 0
            
            llm_hall = llm_only.get("hallucination_rate", 0)
            rag_hall = rag.get("hallucination_rate", 0)
            hall_reduction = ((llm_hall - rag_hall) / llm_hall * 100) if llm_hall > 0 else 0
            
            print(f"\n  {MODELS[key]['name']}:")
            print(f"    Score iyileşme: {improvement:+.1f}%")
            print(f"    Hallucination azalma: {hall_reduction:.1f}%")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Değerlendirme")
    parser.add_argument("--models", nargs="+", default=["openai", "qwen2"], 
                       help="Test edilecek modeller")
    parser.add_argument("--questions", type=int, default=10,
                       help="Soru sayısı")
    parser.add_argument("--no-verbose", "-q", action="store_true",
                       help="Cevapları gizle (sessiz mod)")
    args = parser.parse_args()
    
    # Verbose mod (default: açık, --no-verbose ile kapatılabilir)
    if args.no_verbose:
        globals()['VERBOSE_MODE'] = False
        print("🔇 Sessiz mod - cevaplar gizlenecek\n")
    else:
        print("📝 Verbose mod açık - cevaplar detaylı gösterilecek\n")
    
    run_full_evaluation(args.models, args.questions)

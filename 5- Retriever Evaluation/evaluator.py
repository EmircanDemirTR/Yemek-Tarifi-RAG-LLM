"""
Retriever Değerlendirme Scripti
===============================
Tüm retriever sistemlerini test eder ve karşılaştırır.

Kullanım:
    python evaluator.py                    # Tüm sistemler, k=5
    python evaluator.py --k 10             # k=10 ile test
    python evaluator.py --system bge_m3_wholedoc  # Tek sistem
"""

import os
import sys
import json
import time
import atexit
from datetime import datetime
from pathlib import Path

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

# Qdrant kapanış hatasını bastır
def _cleanup():
    """Program kapanırken Qdrant hatalarını bastır"""
    import sys
    sys.stderr = open(os.devnull, 'w')

atexit.register(_cleanup)

from config import (
    RETRIEVER_SYSTEMS, K_VALUES, DEFAULT_K,
    EVALUATION_SET_PATH, RESULTS_DIR, PROJECT_DIR
)
from metrics import calculate_all_metrics, aggregate_metrics


def load_evaluation_set():
    """Evaluation set'i yükle"""
    with open(EVALUATION_SET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']


def load_retriever(system_key: str):
    """Retriever sistemini yükle"""
    system_info = RETRIEVER_SYSTEMS[system_key]
    system_path = str(system_info['path'])
    
    # Path'i ekle
    original_path = sys.path.copy()
    sys.path.insert(0, system_path)
    
    # Config modülünü temizle (farklı config'ler çakışmasın)
    if 'config' in sys.modules:
        del sys.modules['config']
    if 'searcher' in sys.modules:
        del sys.modules['searcher']
    
    try:
        from searcher import RecipeSearcher
        searcher = RecipeSearcher()
        return searcher, system_info
    finally:
        sys.path = original_path
        if 'config' in sys.modules:
            del sys.modules['config']


def evaluate_system(system_key: str, questions: list, k_values: list = None):
    """Tek bir retriever sistemini değerlendir"""
    k_values = k_values or K_VALUES
    
    print(f"\n{'='*60}")
    print(f"📊 {RETRIEVER_SYSTEMS[system_key]['name']}")
    print(f"{'='*60}")
    
    # Retriever'ı yükle
    print("🔄 Model yükleniyor...")
    searcher, system_info = load_retriever(system_key)
    print(f"✅ Model hazır: {system_info['embedding_model']}")
    
    results = {}
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        all_metrics = []
        
        for i, q in enumerate(questions):
            question = q['question']
            expected = q.get('expected_recipes', [])
            is_impossible = q.get('category') == 'impossible'
            
            # Arama yap
            start_time = time.time()
            search_results = searcher.search(question, top_k=k)
            latency = (time.time() - start_time) * 1000
            
            # Sonuçları al
            retrieved_titles = [r.get('title', '') for r in search_results]
            retrieved_scores = [r.get('score', 0) for r in search_results]
            
            # Metrikleri hesapla
            metrics = calculate_all_metrics(
                retrieved_titles=retrieved_titles,
                expected_titles=expected,
                k=k,
                retrieved_scores=retrieved_scores
            )
            metrics['latency_ms'] = latency
            metrics['question_id'] = q['id']
            
            all_metrics.append(metrics)
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"  İlerleme: {i+1}/{len(questions)}")
        
        # Aggregate
        agg = aggregate_metrics(all_metrics)
        agg['latency_avg_ms'] = sum(m['latency_ms'] for m in all_metrics) / len(all_metrics)
        
        results[f'k={k}'] = {
            'aggregated': agg,
            'detailed': all_metrics
        }
        
        # Özet yazdır
        print(f"  Recall@{k}: {agg.get(f'recall@{k}', 0)*100:.2f}%")
        print(f"  Hit Rate@{k}: {agg.get(f'hit_rate@{k}', 0)*100:.2f}%")
        print(f"  MRR@{k}: {agg.get(f'mrr@{k}', 0):.3f}")
        print(f"  Latency: {agg['latency_avg_ms']:.0f}ms")
    
    return results


def run_full_evaluation(k_values: list = None, systems: list = None):
    """Tüm sistemleri değerlendir"""
    k_values = k_values or K_VALUES
    systems = systems or list(RETRIEVER_SYSTEMS.keys())
    
    print("🚀 Retriever Değerlendirmesi Başlıyor")
    print(f"📝 k değerleri: {k_values}")
    print(f"📦 Sistemler: {systems}")
    
    # Evaluation set yükle
    questions = load_evaluation_set()
    normal_qs = [q for q in questions if q.get('category') != 'impossible']
    impossible_qs = [q for q in questions if q.get('category') == 'impossible']
    
    print(f"\n📊 Evaluation Set:")
    print(f"   Normal sorular: {len(normal_qs)}")
    print(f"   Impossible sorular: {len(impossible_qs)}")
    print(f"   Toplam: {len(questions)}")
    
    all_results = {}
    
    for system_key in systems:
        try:
            results = evaluate_system(system_key, questions, k_values)
            all_results[system_key] = results
        except Exception as e:
            print(f"❌ {system_key} hatası: {e}")
            import traceback
            traceback.print_exc()
    
    # Sonuçları kaydet
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"evaluation_{timestamp}.json"
    
    # Detailed results olmadan kaydet (çok büyük olmasın)
    save_results = {}
    for sys_key, sys_results in all_results.items():
        save_results[sys_key] = {
            k: {'aggregated': v['aggregated']} 
            for k, v in sys_results.items()
        }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Sonuçlar kaydedildi: {result_file}")
    
    # Final karşılaştırma tablosu
    print_comparison_table(all_results, k_values)
    
    return all_results


def print_comparison_table(all_results: dict, k_values: list):
    """Karşılaştırma tablosu yazdır"""
    print("\n" + "="*80)
    print("📊 KARŞILAŞTIRMA TABLOSU")
    print("="*80)
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        print(f"{'Sistem':<30} {'Recall':<10} {'Hit Rate':<10} {'MRR':<10} {'Latency':<10}")
        print("-"*70)
        
        for sys_key, sys_results in all_results.items():
            if f'k={k}' in sys_results:
                agg = sys_results[f'k={k}']['aggregated']
                name = RETRIEVER_SYSTEMS[sys_key]['name']
                recall = agg.get(f'recall@{k}', 0) * 100
                hit_rate = agg.get(f'hit_rate@{k}', 0) * 100
                mrr = agg.get(f'mrr@{k}', 0)
                latency = agg.get('latency_avg_ms', 0)
                
                print(f"{name:<30} {recall:<10.2f}% {hit_rate:<10.2f}% {mrr:<10.3f} {latency:<10.0f}ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Retriever Değerlendirmesi')
    parser.add_argument('--k', type=int, nargs='+', default=K_VALUES,
                        help='Test edilecek k değerleri (örn: --k 5 10)')
    parser.add_argument('--system', type=str, default=None,
                        help='Tek sistem test et (örn: --system bge_m3_wholedoc)')
    
    args = parser.parse_args()
    
    systems = [args.system] if args.system else None
    run_full_evaluation(k_values=args.k, systems=systems)


"""
Retriever Evaluator
===================
Tüm retriever sistemlerini değerlendiren ana modül
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Windows terminal için UTF-8 encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

# TensorFlow uyarılarını sustur
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from config import (
    EVALUATION_SET_PATH,
    RESULTS_DIR,
    RETRIEVER_SYSTEMS,
    K_VALUES
)
from metrics import calculate_all_metrics, aggregate_metrics


class RetrieverEvaluator:
    """Retriever sistemlerini değerlendiren sınıf"""
    
    def __init__(self):
        """Evaluator başlat"""
        self.evaluation_set = self._load_evaluation_set()
        self.results = {}
        
        # Sonuç klasörünü oluştur
        RESULTS_DIR.mkdir(exist_ok=True)
    
    def _load_evaluation_set(self) -> Dict[str, Any]:
        """Evaluation set'i yükle"""
        with open(EVALUATION_SET_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_retriever_system(self, system_key: str):
        """
        Belirtilen retriever sistemini yükle
        
        Returns:
            (searcher, db) tuple
        """
        system_info = RETRIEVER_SYSTEMS[system_key]
        system_path = system_info["path"]
        
        # Sistemi import et
        sys.path.insert(0, str(system_path))
        
        try:
            # Mevcut import'ları temizle
            modules_to_remove = [m for m in sys.modules.keys() 
                               if m in ['config', 'searcher', 'database', 'embedder']]
            for m in modules_to_remove:
                del sys.modules[m]
            
            from searcher import get_searcher
            from database import get_database
            
            searcher = get_searcher()
            db = get_database()
            
            return searcher, db
            
        finally:
            sys.path.remove(str(system_path))
    
    def evaluate_system(
        self, 
        system_key: str, 
        k_values: List[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Tek bir retriever sistemini değerlendir
        
        Args:
            system_key: Sistem anahtarı (bge_m3_wholedoc, e5_large_wholedoc, vb.)
            k_values: Test edilecek k değerleri
            verbose: Detaylı çıktı göster
        
        Returns:
            Değerlendirme sonuçları
        """
        k_values = k_values or K_VALUES
        system_info = RETRIEVER_SYSTEMS[system_key]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 {system_info['name']} Değerlendirmesi")
            print(f"{'='*60}")
            print(f"  Model: {system_info['embedding_model']}")
            print(f"  Chunking: {system_info['chunking']}")
            print(f"  k değerleri: {k_values}")
        
        # Sistemi yükle
        if verbose:
            print(f"\n🔄 Sistem yükleniyor...")
        
        searcher, db = self._load_retriever_system(system_key)
        
        # Her k değeri için sonuçları topla
        all_k_results = {k: [] for k in k_values}
        question_results = []
        
        questions = self.evaluation_set["questions"]
        total_questions = len(questions)
        
        if verbose:
            print(f"📝 {total_questions} soru değerlendirilecek...\n")
        
        total_search_time = 0
        
        impossible_count = 0
        false_positive_count = 0
        
        for idx, q in enumerate(questions, 1):
            question = q["question"]
            expected = q["expected_recipes"]
            is_impossible = q.get("category") == "impossible" or len(expected) == 0
            
            # Arama yap ve süreyi ölç
            start_time = time.time()
            results = searcher.search(question, top_k=max(k_values))
            search_time = time.time() - start_time
            total_search_time += search_time
            
            # Bulunan tarif başlıklarını ve skorları al
            retrieved_titles = [r["title"] for r in results]
            retrieved_scores = [r.get("score", 0) for r in results]
            
            # Her k için metrikleri hesapla
            q_metrics = {
                "question_id": q["id"], 
                "question": question,
                "category": q.get("category", "unknown"),
                "is_impossible": is_impossible
            }
            
            for k in k_values:
                metrics = calculate_all_metrics(
                    retrieved_titles, expected, k, 
                    retrieved_scores, score_threshold=0.5
                )
                q_metrics.update(metrics)
                
                # Impossible olmayan sorular için metrikleri topla
                if not is_impossible:
                    all_k_results[k].append(metrics)
            
            # False Positive kontrolü (impossible sorular için)
            if is_impossible:
                impossible_count += 1
                top_score = retrieved_scores[0] if retrieved_scores else 0
                if top_score >= 0.5:  # Yüksek skor = False Positive
                    false_positive_count += 1
                    q_metrics["false_positive_detail"] = f"Top score: {top_score:.4f} >= 0.5"
                else:
                    q_metrics["false_positive_detail"] = f"Top score: {top_score:.4f} < 0.5 (OK)"
            
            q_metrics["search_time_ms"] = search_time * 1000
            q_metrics["top_score"] = retrieved_scores[0] if retrieved_scores else 0
            q_metrics["retrieved_titles"] = retrieved_titles[:max(k_values)]
            question_results.append(q_metrics)
            
            if verbose and idx % 10 == 0:
                print(f"  ✓ {idx}/{total_questions} soru tamamlandı")
        
        # Metrikleri ortala
        aggregated = {}
        for k in k_values:
            k_aggregated = aggregate_metrics(all_k_results[k])
            aggregated[k] = k_aggregated
        
        # Ortalama arama süresi
        avg_search_time = (total_search_time / total_questions) * 1000  # ms
        
        # False Positive Rate hesapla
        fp_rate = (false_positive_count / impossible_count * 100) if impossible_count > 0 else 0
        
        # Sonuçları derle
        result = {
            "system_key": system_key,
            "system_name": system_info["name"],
            "embedding_model": system_info["embedding_model"],
            "chunking_strategy": system_info["chunking"],
            "evaluation_date": datetime.now().isoformat(),
            "total_questions": total_questions,
            "normal_questions": total_questions - impossible_count,
            "impossible_questions": impossible_count,
            "false_positive_count": false_positive_count,
            "false_positive_rate": fp_rate,
            "k_values": k_values,
            "aggregated_metrics": aggregated,
            "avg_search_time_ms": avg_search_time,
            "question_results": question_results
        }
        
        # Sonuçları göster
        if verbose:
            print(f"\n📈 Sonuçlar (Normal Sorular: {total_questions - impossible_count}):")
            print("-" * 60)
            for k in k_values:
                print(f"\n  k={k}:")
                for metric, value in aggregated[k].items():
                    if not metric.startswith("is_"):
                        print(f"    {metric}: {value:.4f}")
            
            print(f"\n  🎯 Impossible Soru Testi ({impossible_count} soru):")
            print(f"    False Positive: {false_positive_count}/{impossible_count} ({fp_rate:.1f}%)")
            if fp_rate == 0:
                print(f"    ✅ Sistem saçma sorulara düşük skor veriyor!")
            else:
                print(f"    ⚠️  Sistem bazı saçma sorulara yüksek skor verdi")
            
            print(f"\n  ⏱️  Ortalama arama süresi: {avg_search_time:.2f} ms")
        
        # Veritabanı bağlantısını kapat
        db.close()
        
        return result
    
    def evaluate_all_systems(
        self, 
        k_values: List[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Tüm retriever sistemlerini değerlendir
        
        Returns:
            Tüm sistemlerin değerlendirme sonuçları
        """
        k_values = k_values or K_VALUES
        
        print("\n" + "=" * 70)
        print("🚀 TÜM RETRİEVER SİSTEMLERİ DEĞERLENDİRİLİYOR")
        print("=" * 70)
        
        all_results = {}
        
        for system_key in RETRIEVER_SYSTEMS.keys():
            try:
                result = self.evaluate_system(system_key, k_values, verbose)
                all_results[system_key] = result
            except Exception as e:
                print(f"\n❌ {system_key} değerlendirilemedi: {e}")
                all_results[system_key] = {"error": str(e)}
        
        # Özet tablo oluştur
        self._print_comparison_table(all_results, k_values)
        
        # Sonuçları kaydet
        self._save_results(all_results)
        
        return all_results
    
    def _print_comparison_table(self, all_results: Dict, k_values: List[int]):
        """Karşılaştırma tablosunu yazdır"""
        print("\n" + "=" * 100)
        print("📊 KARŞILAŞTIRMA TABLOSU")
        print("=" * 100)
        
        # Başlık satırı
        k = 5  # Ana metrik için k=5 kullan
        header = f"{'Sistem':<28} | Recall@5 | Hit@5  | MRR@5  | FP Rate | Latency"
        print(header)
        print("-" * 100)
        
        # Her sistem için satır
        for system_key, result in all_results.items():
            if "error" in result:
                continue
            
            metrics = result['aggregated_metrics'].get(k, {})
            recall = metrics.get(f'recall@{k}', 0)
            hit = metrics.get(f'hit_rate@{k}', 0)
            mrr = metrics.get(f'mrr@{k}', 0)
            fp_rate = result.get('false_positive_rate', 0)
            latency = result['avg_search_time_ms']
            
            row = f"{result['system_name']:<28} | {recall:>6.2%} | {hit:>5.2%} | {mrr:>5.3f} | {fp_rate:>5.1f}%  | {latency:>5.0f}ms"
            print(row)
        
        print("=" * 100)
        
        # Detaylı k değerleri tablosu
        print("\n📈 Detaylı k Değerleri Karşılaştırması:")
        print("-" * 80)
        for system_key, result in all_results.items():
            if "error" in result:
                continue
            print(f"\n  {result['system_name']}:")
            for k in k_values:
                metrics = result['aggregated_metrics'].get(k, {})
                recall = metrics.get(f'recall@{k}', 0)
                hit = metrics.get(f'hit_rate@{k}', 0)
                print(f"    k={k:>2}: Recall={recall:.2%}, Hit Rate={hit:.2%}")
    
    def _save_results(self, all_results: Dict):
        """Sonuçları dosyaya kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Detaylı JSON
        detailed_path = RESULTS_DIR / f"evaluation_results_{timestamp}.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Sonuçlar kaydedildi: {detailed_path}")
        
        # Özet CSV oluştur
        self._save_summary_csv(all_results, timestamp)
    
    def _save_summary_csv(self, all_results: Dict, timestamp: str):
        """Özet CSV oluştur"""
        csv_path = RESULTS_DIR / f"evaluation_summary_{timestamp}.csv"
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Başlık
            headers = ["System", "Model", "Chunking"]
            for k in K_VALUES:
                headers.extend([f"Recall@{k}", f"HitRate@{k}", f"MRR@{k}", f"Precision@{k}"])
            headers.append("Latency(ms)")
            f.write(",".join(headers) + "\n")
            
            # Veriler
            for system_key, result in all_results.items():
                if "error" in result:
                    continue
                
                row = [
                    result['system_name'],
                    result['embedding_model'],
                    result['chunking_strategy']
                ]
                
                for k in K_VALUES:
                    metrics = result['aggregated_metrics'][k]
                    row.extend([
                        f"{metrics.get(f'recall@{k}', 0):.4f}",
                        f"{metrics.get(f'hit_rate@{k}', 0):.4f}",
                        f"{metrics.get(f'mrr@{k}', 0):.4f}",
                        f"{metrics.get(f'precision@{k}', 0):.4f}"
                    ])
                
                row.append(f"{result['avg_search_time_ms']:.2f}")
                f.write(",".join(row) + "\n")
        
        print(f"💾 Özet CSV: {csv_path}")


def main():
    """Ana fonksiyon"""
    evaluator = RetrieverEvaluator()
    
    print("\n🍳 Yemek Tarifi RAG - Retriever Değerlendirmesi")
    print("=" * 50)
    
    # Tüm sistemleri değerlendir
    results = evaluator.evaluate_all_systems(verbose=True)
    
    print("\n✅ Değerlendirme tamamlandı!")


if __name__ == "__main__":
    main()


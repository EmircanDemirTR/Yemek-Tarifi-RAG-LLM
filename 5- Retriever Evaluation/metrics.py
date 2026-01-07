"""
Retriever Performans Metrikleri
===============================
Recall@k, Hit Rate@k, MRR@k hesaplama fonksiyonları
"""

from typing import List, Dict, Any
import re


def normalize_title(title: str) -> str:
    """
    Tarif başlığını normalize et (karşılaştırma için)
    - Küçük harfe çevir
    - 'Tarifi' kelimesini çıkar
    - Fazla boşlukları temizle
    """
    title = title.lower().strip()
    title = re.sub(r'\s+tarifi?\s*$', '', title)  # Sonundaki "Tarifi" veya "Tarif"
    title = re.sub(r'\s+', ' ', title)  # Çoklu boşlukları tek boşluğa
    return title


def is_relevant(retrieved_title: str, expected_titles: List[str]) -> bool:
    """
    Bulunan tarif başlığının beklenen tariflerden biri olup olmadığını kontrol et
    Fuzzy matching ile benzerlik kontrolü yapar
    """
    retrieved_norm = normalize_title(retrieved_title)
    
    for expected in expected_titles:
        expected_norm = normalize_title(expected)
        
        # Exact match
        if retrieved_norm == expected_norm:
            return True
        
        # Contains match (biri diğerini içeriyor mu)
        if expected_norm in retrieved_norm or retrieved_norm in expected_norm:
            return True
        
        # Kelime bazlı örtüşme (en az %60 kelime eşleşmesi)
        retrieved_words = set(retrieved_norm.split())
        expected_words = set(expected_norm.split())
        
        if len(expected_words) > 0:
            overlap = len(retrieved_words & expected_words)
            overlap_ratio = overlap / len(expected_words)
            if overlap_ratio >= 0.6:
                return True
    
    return False


def recall_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int) -> float:
    """
    Recall@k hesapla
    
    Recall@k = (bulunan benzersiz ilgili doküman sayısı) / (toplam beklenen doküman sayısı)
    
    Args:
        retrieved_titles: Retriever'ın döndürdüğü tarif başlıkları (sıralı)
        expected_titles: Beklenen ilgili tarif başlıkları
        k: Top-k değeri
    
    Returns:
        Recall@k değeri (0-1 arası)
    """
    if not expected_titles:
        return 0.0
    
    top_k_retrieved = retrieved_titles[:k]
    
    # Her expected için bulunan en iyi match'i say (her expected en fazla 1 kez sayılır)
    matched_expected = set()
    
    for retrieved in top_k_retrieved:
        for i, expected in enumerate(expected_titles):
            if i not in matched_expected:
                expected_norm = normalize_title(expected)
                retrieved_norm = normalize_title(retrieved)
                
                # Match kontrolü
                if (expected_norm in retrieved_norm or 
                    retrieved_norm in expected_norm or
                    expected_norm == retrieved_norm):
                    matched_expected.add(i)
                    break
                
                # Kelime bazlı örtüşme
                retrieved_words = set(retrieved_norm.split())
                expected_words = set(expected_norm.split())
                if len(expected_words) > 0:
                    overlap = len(retrieved_words & expected_words)
                    if overlap / len(expected_words) >= 0.6:
                        matched_expected.add(i)
                        break
    
    return len(matched_expected) / len(expected_titles)


def hit_rate_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int) -> float:
    """
    Hit Rate@k hesapla (Binary Recall)
    
    Hit Rate@k = 1 eğer top-k içinde en az bir ilgili doküman varsa, 0 aksi halde
    
    Args:
        retrieved_titles: Retriever'ın döndürdüğü tarif başlıkları (sıralı)
        expected_titles: Beklenen ilgili tarif başlıkları
        k: Top-k değeri
    
    Returns:
        1.0 veya 0.0
    """
    top_k_retrieved = retrieved_titles[:k]
    
    for retrieved in top_k_retrieved:
        if is_relevant(retrieved, expected_titles):
            return 1.0
    
    return 0.0


def mrr_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int) -> float:
    """
    MRR@k (Mean Reciprocal Rank) hesapla
    
    MRR = 1 / (ilk ilgili dokümanın sırası)
    
    Args:
        retrieved_titles: Retriever'ın döndürdüğü tarif başlıkları (sıralı)
        expected_titles: Beklenen ilgili tarif başlıkları
        k: Top-k değeri
    
    Returns:
        Reciprocal Rank değeri (0-1 arası)
    """
    top_k_retrieved = retrieved_titles[:k]
    
    for rank, retrieved in enumerate(top_k_retrieved, 1):
        if is_relevant(retrieved, expected_titles):
            return 1.0 / rank
    
    return 0.0


def precision_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int) -> float:
    """
    Precision@k hesapla
    
    Precision@k = (k içinde bulunan ilgili doküman sayısı) / k
    
    Args:
        retrieved_titles: Retriever'ın döndürdüğü tarif başlıkları (sıralı)
        expected_titles: Beklenen ilgili tarif başlıkları
        k: Top-k değeri
    
    Returns:
        Precision@k değeri (0-1 arası)
    """
    top_k_retrieved = retrieved_titles[:k]
    
    relevant_found = 0
    for retrieved in top_k_retrieved:
        if is_relevant(retrieved, expected_titles):
            relevant_found += 1
    
    return relevant_found / k if k > 0 else 0.0


def false_positive_rate(
    retrieved_titles: List[str],
    retrieved_scores: List[float],
    expected_titles: List[str],
    score_threshold: float = 0.5
) -> float:
    """
    False Positive Rate hesapla (impossible sorular için)
    
    Eğer expected_titles boş ise (impossible soru) ve sistem
    yüksek skorlu sonuç döndürüyorsa bu bir false positive'dir.
    
    Args:
        retrieved_titles: Bulunan tarif başlıkları
        retrieved_scores: Bulunan tariflerin benzerlik skorları
        expected_titles: Beklenen tarifler (impossible için boş)
        score_threshold: Bu skorun üstü "bulundu" sayılır
    
    Returns:
        1.0 eğer false positive (impossible ama yüksek skorlu sonuç var)
        0.0 eğer doğru negatif (impossible ve düşük skorlu/sonuç yok)
    """
    # Bu metrik sadece impossible sorular için anlamlı
    if expected_titles:  # Normal soru, FP hesaplamayı atla
        return -1.0  # Geçersiz işaretle
    
    # Impossible soru - yüksek skorlu sonuç var mı?
    if retrieved_scores and retrieved_scores[0] >= score_threshold:
        return 1.0  # False Positive - olmaması gereken sonuç döndü
    
    return 0.0  # True Negative - doğru şekilde düşük skor


def calculate_all_metrics(
    retrieved_titles: List[str], 
    expected_titles: List[str], 
    k: int,
    retrieved_scores: List[float] = None,
    score_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Tüm metrikleri tek seferde hesapla
    
    Returns:
        {
            "recall@k": float,
            "hit_rate@k": float,
            "mrr@k": float,
            "precision@k": float,
            "is_impossible": bool,
            "false_positive": float (sadece impossible için)
        }
    """
    is_impossible = len(expected_titles) == 0
    
    metrics = {
        f"recall@{k}": recall_at_k(retrieved_titles, expected_titles, k) if not is_impossible else 0.0,
        f"hit_rate@{k}": hit_rate_at_k(retrieved_titles, expected_titles, k) if not is_impossible else 0.0,
        f"mrr@{k}": mrr_at_k(retrieved_titles, expected_titles, k) if not is_impossible else 0.0,
        f"precision@{k}": precision_at_k(retrieved_titles, expected_titles, k) if not is_impossible else 0.0,
        "is_impossible": 1.0 if is_impossible else 0.0
    }
    
    # Impossible sorular için False Positive Rate
    if is_impossible and retrieved_scores:
        metrics["false_positive"] = false_positive_rate(
            retrieved_titles, retrieved_scores, expected_titles, score_threshold
        )
    
    return metrics


def aggregate_metrics(all_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Tüm sorular için metrikleri ortala
    
    Args:
        all_results: Her soru için hesaplanan metrikler listesi
    
    Returns:
        Ortalama metrikler
    """
    if not all_results:
        return {}
    
    aggregated = {}
    metric_keys = all_results[0].keys()
    
    for key in metric_keys:
        values = [r[key] for r in all_results if r.get(key) is not None]
        if values:
            aggregated[key] = sum(values) / len(values)
    
    return aggregated


if __name__ == "__main__":
    # Test
    retrieved = ["Mercimek Çorbası Tarifi", "Kırmızı Mercimek Çorbası", "Ezogelin Çorbası", "Domates Çorbası", "Tarhana Çorbası"]
    expected = ["Mercimek Çorbası", "Kırmızı Mercimek Çorbası", "Süzme Mercimek Çorbası"]
    
    print("Test Sonuçları:")
    print(f"Retrieved: {retrieved}")
    print(f"Expected: {expected}")
    print()
    
    for k in [1, 3, 5]:
        metrics = calculate_all_metrics(retrieved, expected, k)
        print(f"k={k}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()


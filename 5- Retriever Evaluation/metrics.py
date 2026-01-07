"""
Retriever Değerlendirme Metrikleri
==================================
Recall@k, Hit Rate@k, MRR@k, Precision@k, False Positive Rate
"""

from typing import List, Dict


def recall_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int) -> float:
    """
    Recall@k hesapla
    
    Recall@k = (Top-k içinde bulunan beklenen doküman sayısı) / (Toplam beklenen doküman sayısı)
    """
    if not expected_titles:
        return 0.0
    
    top_k = retrieved_titles[:k]
    
    # Normalize edilmiş karşılaştırma
    top_k_normalized = [t.lower().strip() for t in top_k]
    expected_normalized = [t.lower().strip() for t in expected_titles]
    
    # Bulunan benzersiz ilgili doküman sayısı
    found_count = 0
    for expected in expected_normalized:
        for retrieved in top_k_normalized:
            if expected in retrieved or retrieved in expected:
                found_count += 1
                break
    
    return found_count / len(expected_titles)


def hit_rate_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int) -> float:
    """
    Hit Rate@k (Success Rate) hesapla
    
    En az bir ilgili doküman bulunduysa 1, aksi halde 0
    """
    if not expected_titles:
        return 0.0
    
    top_k = retrieved_titles[:k]
    
    # Normalize edilmiş karşılaştırma
    top_k_normalized = [t.lower().strip() for t in top_k]
    expected_normalized = [t.lower().strip() for t in expected_titles]
    
    for expected in expected_normalized:
        for retrieved in top_k_normalized:
            if expected in retrieved or retrieved in expected:
                return 1.0
    
    return 0.0


def mrr_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int) -> float:
    """
    Mean Reciprocal Rank@k hesapla
    
    İlk ilgili dokümanın sıralamasının tersi
    """
    if not expected_titles:
        return 0.0
    
    top_k = retrieved_titles[:k]
    
    # Normalize edilmiş karşılaştırma
    top_k_normalized = [t.lower().strip() for t in top_k]
    expected_normalized = [t.lower().strip() for t in expected_titles]
    
    for i, retrieved in enumerate(top_k_normalized, 1):
        for expected in expected_normalized:
            if expected in retrieved or retrieved in expected:
                return 1.0 / i
    
    return 0.0


def precision_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int) -> float:
    """
    Precision@k hesapla
    
    Precision@k = (Top-k içindeki ilgili doküman sayısı) / k
    """
    if not expected_titles or k == 0:
        return 0.0
    
    top_k = retrieved_titles[:k]
    
    # Normalize edilmiş karşılaştırma
    top_k_normalized = [t.lower().strip() for t in top_k]
    expected_normalized = [t.lower().strip() for t in expected_titles]
    
    relevant_count = 0
    for retrieved in top_k_normalized:
        for expected in expected_normalized:
            if expected in retrieved or retrieved in expected:
                relevant_count += 1
                break
    
    return relevant_count / k


def false_positive_rate(
    retrieved_titles: List[str],
    retrieved_scores: List[float],
    expected_titles: List[str],
    score_threshold: float = 0.5
) -> float:
    """
    False Positive Rate hesapla (impossible sorular için)
    """
    if expected_titles:
        return -1.0
    
    if retrieved_scores and retrieved_scores[0] >= score_threshold:
        return 1.0
    
    return 0.0


def calculate_all_metrics(
    retrieved_titles: List[str], 
    expected_titles: List[str], 
    k: int,
    retrieved_scores: List[float] = None,
    score_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Tüm metrikleri tek seferde hesapla
    """
    is_impossible = len(expected_titles) == 0
    
    metrics = {
        f"recall@{k}": recall_at_k(retrieved_titles, expected_titles, k) if not is_impossible else 0.0,
        f"hit_rate@{k}": hit_rate_at_k(retrieved_titles, expected_titles, k) if not is_impossible else 0.0,
        f"mrr@{k}": mrr_at_k(retrieved_titles, expected_titles, k) if not is_impossible else 0.0,
        f"precision@{k}": precision_at_k(retrieved_titles, expected_titles, k) if not is_impossible else 0.0,
        "is_impossible": 1.0 if is_impossible else 0.0
    }
    
    if is_impossible and retrieved_scores:
        metrics["false_positive"] = false_positive_rate(
            retrieved_titles, retrieved_scores, expected_titles, score_threshold
        )
    
    return metrics


def aggregate_metrics(all_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Tüm sorular için metrikleri ortala
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


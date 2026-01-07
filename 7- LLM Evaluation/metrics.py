"""
LLM Değerlendirme Metrikleri
EM, F1-score, Keyword Match, Hallucination Rate
"""
import re
from typing import List, Tuple, Dict


def normalize_text(text: str) -> str:
    """Metni normalize et"""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ]', '', text)
    return text


def get_tokens(text: str) -> List[str]:
    """Metni tokenlara ayır"""
    return normalize_text(text).split()


def exact_match(prediction: str, gold: str) -> float:
    """Exact Match - tam eşleşme"""
    if not gold:
        return 0.0
    
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    
    if pred_norm == gold_norm:
        return 1.0
    if gold_norm in pred_norm:
        return 0.5
    return 0.0


def f1_score(prediction: str, gold: str) -> Tuple[float, float, float]:
    """Token bazlı F1-score"""
    if not gold:
        return 0.0, 0.0, 0.0
    
    pred_tokens = set(get_tokens(prediction))
    gold_tokens = set(get_tokens(gold))
    
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    
    common = pred_tokens.intersection(gold_tokens)
    
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def keyword_match_score(prediction: str, keywords: List[str]) -> float:
    """
    Anahtar kelime eşleşme skoru
    Prediction içinde kaç keyword bulunduğunu ölçer
    """
    if not keywords:
        return 0.0
    
    pred_lower = normalize_text(prediction)
    found = 0
    
    for keyword in keywords:
        keyword_norm = normalize_text(keyword)
        if keyword_norm in pred_lower:
            found += 1
    
    return found / len(keywords)


def recipe_match_score(prediction: str, expected_recipes: List[str]) -> float:
    """
    Beklenen tarif isimlerinin prediction'da geçip geçmediğini kontrol et
    """
    if not expected_recipes:
        return 0.0
    
    pred_lower = normalize_text(prediction)
    found = 0
    
    for recipe in expected_recipes:
        recipe_norm = normalize_text(recipe)
        # Tarif isminin herhangi bir kısmı geçiyor mu
        recipe_words = recipe_norm.split()
        if any(word in pred_lower for word in recipe_words if len(word) > 2):
            found += 1
    
    return found / len(expected_recipes)


def detect_hallucination(prediction: str, context: str = None, gold: str = None, keywords: List[str] = None) -> Tuple[bool, str]:
    """
    Hallucination tespiti - geliştirilmiş versiyon
    """
    pred_lower = prediction.lower()
    
    # "Bulunamadı" türü cevaplar
    not_found_phrases = [
        "bulunamadı", "bulunmuyor", "bilmiyorum", "bilinmiyor",
        "veritabanımda yok", "mevcut değil", "tarif yok",
        "bu tarif", "böyle bir tarif"
    ]
    
    for phrase in not_found_phrases:
        if phrase in pred_lower:
            return False, "not_found_response"
    
    # Keyword kontrolü
    if keywords:
        kw_score = keyword_match_score(prediction, keywords)
        if kw_score >= 0.3:  # En az %30 keyword eşleşmesi
            return False, "keyword_match"
    
    # Context kontrolü (RAG modu)
    if context:
        context_lower = context.lower()
        pred_tokens = get_tokens(prediction)
        important_words = [t for t in pred_tokens if len(t) > 4]
        
        if important_words:
            found_in_context = sum(1 for w in important_words if w in context_lower)
            if found_in_context >= len(important_words) * 0.3:
                return False, "context_match"
    
    # Gold answer kontrolü
    if gold:
        _, _, f1 = f1_score(prediction, gold)
        if f1 >= 0.15:
            return False, "f1_match"
        
        # Keyword match de kontrol et
        if keywords:
            kw_score = keyword_match_score(prediction, keywords)
            if kw_score >= 0.2:
                return False, "partial_keyword_match"
    
    # Çok kısa veya boş cevap
    if len(prediction.strip()) < 20:
        return True, "too_short"
    
    return True, "no_evidence"


def context_faithfulness_score(prediction: str, context: str) -> float:
    """
    Cevabın context'e ne kadar sadık olduğunu ölç
    RAG değerlendirmesi için önemli
    """
    if not context:
        return 0.0
    
    pred_tokens = get_tokens(prediction)
    context_lower = normalize_text(context)
    
    # Önemli kelimeler (4+ karakter)
    important_words = [t for t in pred_tokens if len(t) > 3]
    
    if not important_words:
        return 0.0
    
    found_in_context = sum(1 for w in important_words if w in context_lower)
    return found_in_context / len(important_words)


def calculate_llm_metrics(prediction: str, gold: str, context: str = None, 
                          keywords: List[str] = None, expected_recipes: List[str] = None) -> Dict:
    """
    Tüm LLM metriklerini hesapla
    RAG ve LLM-Only için farklı ağırlıklandırma
    """
    em = exact_match(prediction, gold)
    precision, recall, f1 = f1_score(prediction, gold)
    
    # Keyword ve recipe match
    kw_score = keyword_match_score(prediction, keywords) if keywords else 0.0
    recipe_score = recipe_match_score(prediction, expected_recipes) if expected_recipes else 0.0
    
    # Context faithfulness (RAG için)
    faithfulness = context_faithfulness_score(prediction, context) if context else 0.0
    
    # Hallucination
    is_hall, hall_reason = detect_hallucination(prediction, context, gold, keywords)
    
    # Combined score - RAG ve LLM-Only için farklı
    if context:
        # RAG modu: faithfulness + keyword + recipe ağırlıklı
        combined_score = (kw_score * 0.4 + recipe_score * 0.3 + faithfulness * 0.2 + f1 * 0.1)
    else:
        # LLM-Only modu: keyword + f1 ağırlıklı
        combined_score = (kw_score * 0.5 + f1 * 0.5) if kw_score > 0 else f1
    
    return {
        "exact_match": em,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "keyword_score": kw_score,
        "recipe_score": recipe_score,
        "faithfulness": faithfulness,
        "combined_score": combined_score,
        "is_hallucination": is_hall,
        "hallucination_reason": hall_reason
    }


def aggregate_llm_metrics(all_results: List[Dict]) -> Dict:
    """Tüm sonuçları ortala"""
    if not all_results:
        return {}
    
    total = len(all_results)
    
    return {
        "exact_match": sum(r.get("exact_match", 0) for r in all_results) / total,
        "precision": sum(r.get("precision", 0) for r in all_results) / total,
        "recall": sum(r.get("recall", 0) for r in all_results) / total,
        "f1": sum(r.get("f1", 0) for r in all_results) / total,
        "keyword_score": sum(r.get("keyword_score", 0) for r in all_results) / total,
        "recipe_score": sum(r.get("recipe_score", 0) for r in all_results) / total,
        "faithfulness": sum(r.get("faithfulness", 0) for r in all_results) / total,
        "combined_score": sum(r.get("combined_score", 0) for r in all_results) / total,
        "hallucination_rate": sum(1 for r in all_results if r.get("is_hallucination", False)) / total,
        "total_questions": total
    }


if __name__ == "__main__":
    # Test
    gold = "Mercimek çorbası tarifi: 1 su bardağı kırmızı mercimek yıkanır. Soğan, havuç ve patates doğranır."
    pred = "Mercimek çorbası yapmak için önce mercimeği yıkayın. Soğan ve havuç ekleyin, pişirin."
    keywords = ["mercimek", "çorba", "soğan", "havuç", "yıka"]
    
    result = calculate_llm_metrics(pred, gold, keywords=keywords)
    print("Test Sonuçları:")
    for k, v in result.items():
        print(f"  {k}: {v}")

"""
Prompt Şablonları
"""

# RAG modu için prompt (context ile) - daha esnek
RAG_PROMPT = """Sen bir Türk mutfağı uzmanısın. Aşağıdaki tarif veritabanından bulunan bilgileri kullanarak soruyu cevapla.

VERİTABANINDAN BULUNAN TARİFLER:
{context}

SORU: {question}

TALİMATLAR:
- Yukarıdaki tarif bilgilerini temel al
- Tariflerdeki malzemeleri ve yapılışı özetle
- Türkçe ve açıklayıcı cevap ver
- Birden fazla tarif varsa en uygun olanı öner

CEVAP:"""

# LLM-Only modu için prompt (context olmadan)
LLM_ONLY_PROMPT = """Sen bir Türk mutfağı uzmanısın.

SORU: {question}

TALİMATLAR:
- Türkçe ve kısa cevap ver
- Tarif sorusu ise malzemeleri ve yapılışı özetle
- Emin olmadığın bilgileri uydurma

CEVAP:"""


def format_context(recipes: list, max_length: int = 3000) -> str:
    """Retriever sonuçlarını context formatına çevir - genişletilmiş"""
    context_parts = []
    total_length = 0
    
    for i, recipe in enumerate(recipes, 1):
        title = recipe.get("title", "Bilinmeyen Tarif")
        ingredients = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", [])
        
        # Tüm malzemeleri al
        ing_text = ", ".join(ingredients) if ingredients else "Belirtilmemiş"
        
        # Daha fazla talimat al
        inst_text = " ".join(instructions[:5]) if instructions else "Belirtilmemiş"
        if len(inst_text) > 400:
            inst_text = inst_text[:400] + "..."
        
        part = f"[{i}] {title}\nMalzemeler: {ing_text}\nYapılış: {inst_text}\n"
        
        if total_length + len(part) > max_length:
            break
        
        context_parts.append(part)
        total_length += len(part)
    
    return "\n".join(context_parts)


def create_rag_prompt(question: str, recipes: list) -> str:
    """RAG promptu oluştur"""
    context = format_context(recipes)
    return RAG_PROMPT.format(context=context, question=question)


def create_llm_only_prompt(question: str) -> str:
    """LLM-Only promptu oluştur"""
    return LLM_ONLY_PROMPT.format(question=question)

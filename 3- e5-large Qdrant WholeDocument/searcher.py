"""
Tarif Arama Modülü
==================
Kullanıcı sorgularına göre tarif arama (E5-Large)
"""

from typing import List, Dict, Any, Optional
from config import DEFAULT_TOP_K, SCORE_THRESHOLD
from embedder import get_embedder
from database import get_database


class RecipeSearcher:
    """Tarif arama sınıfı"""
    
    def __init__(self):
        """Embedder ve database bağlantılarını başlat"""
        self.embedder = get_embedder()
        self.db = get_database()
    
    def search(
        self, 
        query: str, 
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = None,
        ingredient_filter: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tarif ara
        
        Args:
            query: Kullanıcı sorgusu (örn: "tavuklu makarna", "elimde patates var")
            top_k: Döndürülecek sonuç sayısı
            score_threshold: Minimum benzerlik skoru
            ingredient_filter: Belirli malzemeleri içeren tarifleri filtrele
        
        Returns:
            Bulunan tarifler listesi
        """
        # Sorguyu vektöre dönüştür (query prefix otomatik eklenir)
        query_vector = self.embedder.embed_query(query)
        
        # Veritabanında ara
        results = self.db.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold or SCORE_THRESHOLD,
            ingredient_filter=ingredient_filter
        )
        
        return results
    
    def search_by_ingredients(
        self, 
        ingredients: List[str], 
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Malzeme listesine göre tarif ara
        
        Args:
            ingredients: Malzeme listesi (örn: ["tavuk", "patates", "biber"])
            top_k: Döndürülecek sonuç sayısı
        
        Returns:
            Bulunan tarifler listesi
        """
        # Malzemeleri sorgu olarak birleştir
        query = f"Elimde şu malzemeler var: {', '.join(ingredients)}. Bu malzemelerle yapılabilecek tarif"
        
        return self.search(
            query=query,
            top_k=top_k,
            ingredient_filter=ingredients
        )
    
    def search_recipe_by_name(
        self, 
        recipe_name: str, 
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Tarif adına göre ara
        
        Args:
            recipe_name: Tarif adı (örn: "karnıyarık", "mercimek çorbası")
            top_k: Döndürülecek sonuç sayısı
        
        Returns:
            Bulunan tarifler listesi
        """
        query = f"{recipe_name} tarifi nasıl yapılır"
        return self.search(query=query, top_k=top_k)
    
    def get_similar_recipes(
        self, 
        recipe_id: int, 
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Benzer tarifleri bul
        
        Args:
            recipe_id: Referans tarif ID'si
            top_k: Döndürülecek sonuç sayısı
        
        Returns:
            Benzer tarifler listesi
        """
        # Referans tarifi getir
        reference = self.db.get_recipe_by_id(recipe_id)
        if not reference:
            return []
        
        # Tarif metnini oluştur ve ara (passage prefix ile)
        text = self.embedder.create_recipe_text(reference, add_prefix=True)
        query_vector = self.embedder.embed_single(text)
        
        # +1 çünkü kendisi de sonuçlarda olacak, onu çıkaracağız
        results = self.db.search(query_vector=query_vector, top_k=top_k + 1)
        
        # Kendisini çıkar
        return [r for r in results if r['id'] != recipe_id][:top_k]


def format_recipe_result(recipe: Dict[str, Any], show_instructions: bool = False) -> str:
    """
    Tarif sonucunu güzel formatla
    
    Args:
        recipe: Tarif dictionary
        show_instructions: Talimatları göster
    
    Returns:
        Formatlanmış string
    """
    output = []
    output.append(f"📗 {recipe['title']}")
    output.append(f"   Skor: {recipe.get('score', 0):.4f}")
    output.append(f"   🔗 {recipe.get('url', 'N/A')}")
    
    # Malzemeler
    ingredients = recipe.get('ingredients', [])
    output.append(f"   📦 Malzemeler ({len(ingredients)} adet):")
    for ing in ingredients[:5]:  # İlk 5 malzeme
        output.append(f"      • {ing}")
    if len(ingredients) > 5:
        output.append(f"      ... ve {len(ingredients) - 5} malzeme daha")
    
    # Talimatlar (isteğe bağlı)
    if show_instructions:
        instructions = recipe.get('instructions', [])
        output.append(f"   📝 Yapılışı ({len(instructions)} adım):")
        for i, step in enumerate(instructions, 1):
            output.append(f"      {i}. {step[:100]}{'...' if len(step) > 100 else ''}")
    
    return "\n".join(output)


def format_search_results(
    results: List[Dict[str, Any]], 
    show_instructions: bool = False
) -> str:
    """Arama sonuçlarını formatla"""
    if not results:
        return "❌ Sonuç bulunamadı."
    
    output = [f"🔍 {len(results)} tarif bulundu:\n"]
    output.append("=" * 60)
    
    for i, recipe in enumerate(results, 1):
        output.append(f"\n[{i}] {format_recipe_result(recipe, show_instructions)}")
        output.append("-" * 60)
    
    return "\n".join(output)


# Singleton instance
_searcher_instance = None

def get_searcher() -> RecipeSearcher:
    """Searcher singleton instance döndür"""
    global _searcher_instance
    if _searcher_instance is None:
        _searcher_instance = RecipeSearcher()
    return _searcher_instance


if __name__ == "__main__":
    # Test aramaları
    searcher = get_searcher()
    
    print("\n" + "=" * 60)
    print("🧪 ARAMA TESTLERİ (E5-Large)")
    print("=" * 60)
    
    # Test 1: Genel arama
    print("\n📝 Test 1: 'tavuklu makarna' araması")
    results = searcher.search("tavuklu makarna", top_k=3)
    print(format_search_results(results))
    
    # Test 2: Malzeme bazlı arama
    print("\n📝 Test 2: Malzeme bazlı arama (patates, soğan)")
    results = searcher.search_by_ingredients(["patates", "soğan"], top_k=3)
    print(format_search_results(results))
    
    # Test 3: Tarif adı araması
    print("\n📝 Test 3: 'mercimek çorbası' tarif araması")
    results = searcher.search_recipe_by_name("mercimek çorbası", top_k=3)
    print(format_search_results(results, show_instructions=True))


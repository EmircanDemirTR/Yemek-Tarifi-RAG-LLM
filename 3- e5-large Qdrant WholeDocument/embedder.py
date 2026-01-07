"""
E5-Large Embedding Modülü
=========================
Tarif metinlerini vektörlere dönüştürme işlemleri
(sentence-transformers ile)

NOT: E5 modeli query ve passage için prefix kullanır!
- Query: "query: ..."
- Passage/Document: "passage: ..."
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, BATCH_SIZE, QUERY_PREFIX, PASSAGE_PREFIX


class RecipeEmbedder:
    """E5-Large ile tarif embedding işlemleri"""
    
    def __init__(self):
        """Model yükle"""
        print(f"🔄 E5-Large modeli yükleniyor: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        print("✅ Model başarıyla yüklendi!")
        print(f"📊 Embedding boyutu: {self.model.get_sentence_embedding_dimension()}")
    
    def create_recipe_text(self, recipe: Dict[str, Any], add_prefix: bool = True) -> str:
        """
        Tarif verisinden embedding için metin oluştur
        
        Args:
            recipe: Tarif dictionary
            add_prefix: E5 passage prefix eklensin mi
        
        Format:
        - Başlık öne çıkarılır
        - Malzemeler virgülle ayrılmış liste
        - Talimatlar paragraf olarak
        """
        title = recipe.get("title", "")
        ingredients = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", [])
        
        # Malzemeleri temizle ve birleştir
        ingredients_text = ", ".join(ingredients)
        
        # Talimatları birleştir
        instructions_text = " ".join(instructions)
        
        # Final metin
        text = f"""Tarif: {title}

Malzemeler: {ingredients_text}

Yapılışı: {instructions_text}"""
        
        # E5 modeli için passage prefix ekle
        if add_prefix:
            text = f"{PASSAGE_PREFIX}{text}"
        
        return text
    
    def embed_single(self, text: str) -> List[float]:
        """Tek bir metni vektöre dönüştür"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Birden fazla metni vektörlere dönüştür"""
        embeddings = self.model.encode(
            texts, 
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return [emb.tolist() for emb in embeddings]
    
    def embed_recipe(self, recipe: Dict[str, Any]) -> List[float]:
        """Tek bir tarifi vektöre dönüştür (passage prefix ile)"""
        text = self.create_recipe_text(recipe, add_prefix=True)
        return self.embed_single(text)
    
    def embed_recipes(self, recipes: List[Dict[str, Any]]) -> List[List[float]]:
        """Birden fazla tarifi vektörlere dönüştür (passage prefix ile)"""
        texts = [self.create_recipe_text(r, add_prefix=True) for r in recipes]
        return self.embed_batch(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Kullanıcı sorgusunu vektöre dönüştür (query prefix ile)
        
        E5 modeli için sorguların başına "query: " eklenir
        """
        query_with_prefix = f"{QUERY_PREFIX}{query}"
        return self.embed_single(query_with_prefix)
    
    def get_embedding_dimension(self) -> int:
        """Embedding boyutunu döndür"""
        return self.model.get_sentence_embedding_dimension()


# Singleton instance
_embedder_instance = None

def get_embedder() -> RecipeEmbedder:
    """Embedder singleton instance döndür"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = RecipeEmbedder()
    return _embedder_instance


if __name__ == "__main__":
    # Test
    embedder = get_embedder()
    
    test_recipe = {
        "title": "Test Tarifi",
        "ingredients": ["un", "şeker", "yumurta"],
        "instructions": ["Malzemeleri karıştır.", "Fırında pişir."]
    }
    
    text = embedder.create_recipe_text(test_recipe)
    print("📝 Oluşturulan metin (passage prefix ile):")
    print(text)
    print()
    
    vector = embedder.embed_recipe(test_recipe)
    print(f"📊 Vektör boyutu: {len(vector)}")
    print(f"📊 İlk 5 değer: {vector[:5]}")
    
    # Query testi
    print("\n📝 Query embedding testi:")
    query = "tavuklu makarna"
    query_vector = embedder.embed_query(query)
    print(f"Query: '{QUERY_PREFIX}{query}'")
    print(f"📊 Vektör boyutu: {len(query_vector)}")


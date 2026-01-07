"""
BGE-M3 Embedding Modülü
=======================
Tarif metinlerini vektörlere dönüştürme işlemleri
(sentence-transformers ile)
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, BATCH_SIZE


class RecipeEmbedder:
    """BGE-M3 ile tarif embedding işlemleri"""
    
    def __init__(self):
        """Model yükle"""
        print(f"🔄 BGE-M3 modeli yükleniyor: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        print("✅ Model başarıyla yüklendi!")
        print(f"📊 Embedding boyutu: {self.model.get_sentence_embedding_dimension()}")
    
    def create_recipe_text(self, recipe: Dict[str, Any]) -> str:
        """
        Tarif verisinden embedding için metin oluştur
        
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
        """Tek bir tarifi vektöre dönüştür"""
        text = self.create_recipe_text(recipe)
        return self.embed_single(text)
    
    def embed_recipes(self, recipes: List[Dict[str, Any]]) -> List[List[float]]:
        """Birden fazla tarifi vektörlere dönüştür"""
        texts = [self.create_recipe_text(r) for r in recipes]
        return self.embed_batch(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """Kullanıcı sorgusunu vektöre dönüştür"""
        return self.embed_single(query)
    
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
    print("📝 Oluşturulan metin:")
    print(text)
    print()
    
    vector = embedder.embed_recipe(test_recipe)
    print(f"📊 Vektör boyutu: {len(vector)}")
    print(f"📊 İlk 5 değer: {vector[:5]}")

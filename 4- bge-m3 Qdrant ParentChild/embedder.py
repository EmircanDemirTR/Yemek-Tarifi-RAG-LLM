"""
BGE-M3 Embedding Modülü - Parent-Child Chunking
================================================
Tarif metinlerini chunk'lara bölüp vektörlere dönüştürme
"""

from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from config import (
    MODEL_NAME, 
    BATCH_SIZE,
    CHUNK_TYPE_INGREDIENTS,
    CHUNK_TYPE_INSTRUCTIONS
)


class RecipeEmbedder:
    """BGE-M3 ile tarif embedding işlemleri (Parent-Child)"""
    
    def __init__(self):
        """Model yükle"""
        print(f"🔄 BGE-M3 modeli yükleniyor: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        print("✅ Model başarıyla yüklendi!")
        print(f"📊 Embedding boyutu: {self.model.get_sentence_embedding_dimension()}")
    
    # =========================================================================
    # CHUNK OLUŞTURMA
    # =========================================================================
    
    def create_ingredients_chunk(self, recipe: Dict[str, Any]) -> str:
        """
        Malzeme chunk'ı oluştur (Başlık + Malzemeler)
        
        Bu chunk, malzeme bazlı aramalarda eşleşir:
        - "elimde tavuk ve patates var"
        - "domates, biber, patlıcan ile ne yapabilirim"
        """
        title = recipe.get("title", "")
        ingredients = recipe.get("ingredients", [])
        
        ingredients_text = ", ".join(ingredients)
        
        return f"""Tarif: {title}

Malzemeler: {ingredients_text}"""
    
    def create_instructions_chunk(self, recipe: Dict[str, Any]) -> str:
        """
        Talimat chunk'ı oluştur (Başlık + Yapılış)
        
        Bu chunk, yapılış/yöntem bazlı aramalarda eşleşir:
        - "fırında nasıl pişirilir"
        - "kaç dakika kaynatılmalı"
        """
        title = recipe.get("title", "")
        instructions = recipe.get("instructions", [])
        
        instructions_text = " ".join(instructions)
        
        return f"""Tarif: {title}

Yapılışı: {instructions_text}"""
    
    def create_full_text(self, recipe: Dict[str, Any]) -> str:
        """
        Tam tarif metni oluştur (Parent - sadece payload için)
        """
        title = recipe.get("title", "")
        ingredients = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", [])
        
        ingredients_text = ", ".join(ingredients)
        instructions_text = " ".join(instructions)
        
        return f"""Tarif: {title}

Malzemeler: {ingredients_text}

Yapılışı: {instructions_text}"""
    
    def create_chunks(self, recipe: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Tarif için tüm chunk'ları oluştur
        
        Returns:
            List of (chunk_type, chunk_text) tuples
        """
        return [
            (CHUNK_TYPE_INGREDIENTS, self.create_ingredients_chunk(recipe)),
            (CHUNK_TYPE_INSTRUCTIONS, self.create_instructions_chunk(recipe))
        ]
    
    # =========================================================================
    # EMBEDDING
    # =========================================================================
    
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
    
    def embed_recipe_chunks(self, recipe: Dict[str, Any]) -> List[Tuple[str, List[float]]]:
        """
        Tek bir tarifin tüm chunk'larını embed et
        
        Returns:
            List of (chunk_type, embedding) tuples
        """
        chunks = self.create_chunks(recipe)
        result = []
        
        for chunk_type, chunk_text in chunks:
            embedding = self.embed_single(chunk_text)
            result.append((chunk_type, embedding))
        
        return result
    
    def embed_recipes_chunks(
        self, 
        recipes: List[Dict[str, Any]]
    ) -> List[List[Tuple[str, List[float]]]]:
        """
        Birden fazla tarifin tüm chunk'larını embed et
        
        Returns:
            List of recipes, each containing list of (chunk_type, embedding)
        """
        # Tüm chunk'ları topla
        all_chunks = []
        chunk_mapping = []  # (recipe_idx, chunk_type)
        
        for recipe_idx, recipe in enumerate(recipes):
            chunks = self.create_chunks(recipe)
            for chunk_type, chunk_text in chunks:
                all_chunks.append(chunk_text)
                chunk_mapping.append((recipe_idx, chunk_type))
        
        # Toplu embedding
        all_embeddings = self.embed_batch(all_chunks)
        
        # Sonuçları tariflere göre grupla
        results = [[] for _ in recipes]
        
        for embedding, (recipe_idx, chunk_type) in zip(all_embeddings, chunk_mapping):
            results[recipe_idx].append((chunk_type, embedding))
        
        return results
    
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
        "title": "Tavuklu Makarna",
        "ingredients": ["makarna", "tavuk göğsü", "domates sosu", "soğan", "sarımsak"],
        "instructions": [
            "Tavukları küp küp doğrayın.", 
            "Soğan ve sarımsağı kavurun.",
            "Tavukları ekleyip soteleyin.",
            "Domates sosunu ekleyin.",
            "Haşlanmış makarnayı ekleyip karıştırın."
        ]
    }
    
    print("📋 Test tarifi:", test_recipe["title"])
    print()
    
    # Chunk'ları göster
    print("=" * 60)
    print("📦 OLUŞTURULAN CHUNK'LAR")
    print("=" * 60)
    
    chunks = embedder.create_chunks(test_recipe)
    for chunk_type, chunk_text in chunks:
        print(f"\n[{chunk_type.upper()}]")
        print("-" * 40)
        print(chunk_text)
    
    # Embedding boyutları
    print("\n" + "=" * 60)
    print("📊 EMBEDDING BİLGİLERİ")
    print("=" * 60)
    
    embedded_chunks = embedder.embed_recipe_chunks(test_recipe)
    for chunk_type, embedding in embedded_chunks:
        print(f"  {chunk_type}: {len(embedding)} boyut")


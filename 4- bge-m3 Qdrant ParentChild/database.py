"""
Qdrant Veritabanı Modülü - Parent-Child Chunking
=================================================
Vektör veritabanı işlemleri (Parent-Child desteği ile)
"""

import sys
import os

# Windows terminal için UTF-8 encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

from typing import List, Dict, Any, Optional
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText
)
from config import (
    QDRANT_PATH, 
    COLLECTION_NAME, 
    EMBEDDING_DIM, 
    DISTANCE_METRIC,
    INDEX_BATCH_SIZE,
    CHUNK_TYPE_INGREDIENTS,
    CHUNK_TYPE_INSTRUCTIONS,
    CHUNKS_PER_RECIPE
)


class RecipeDatabase:
    """Qdrant vektör veritabanı işlemleri (Parent-Child)"""
    
    def __init__(self):
        """Veritabanı bağlantısı oluştur"""
        print(f"🔄 Qdrant veritabanına bağlanılıyor: {QDRANT_PATH}")
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        self.client = QdrantClient(path=str(QDRANT_PATH))
        print("✅ Veritabanı bağlantısı başarılı!")
    
    def close(self):
        """Veritabanı bağlantısını kapat"""
        try:
            if hasattr(self, 'client') and self.client is not None:
                self.client.close()
        except Exception:
            pass  # Kapanış hatalarını yoksay
    
    def collection_exists(self) -> bool:
        """Collection var mı kontrol et"""
        collections = self.client.get_collections().collections
        return any(c.name == COLLECTION_NAME for c in collections)
    
    def create_collection(self, recreate: bool = False):
        """
        Collection oluştur
        
        Args:
            recreate: True ise mevcut collection silinip yeniden oluşturulur
        """
        if self.collection_exists():
            if recreate:
                print(f"🗑️  Mevcut collection siliniyor: {COLLECTION_NAME}")
                self.client.delete_collection(COLLECTION_NAME)
            else:
                print(f"ℹ️  Collection zaten mevcut: {COLLECTION_NAME}")
                return
        
        # Distance metric mapping
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT
        }
        
        print(f"📦 Collection oluşturuluyor: {COLLECTION_NAME}")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=distance_map.get(DISTANCE_METRIC, Distance.COSINE)
            )
        )
        print("✅ Collection başarıyla oluşturuldu!")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Collection bilgilerini getir"""
        if not self.collection_exists():
            return {"exists": False}
        
        info = self.client.get_collection(COLLECTION_NAME)
        
        # Chunk sayısından tarif sayısını hesapla
        chunks_count = info.points_count
        recipes_count = chunks_count // CHUNKS_PER_RECIPE if chunks_count else 0
        
        return {
            "exists": True,
            "points_count": chunks_count,
            "recipes_count": recipes_count,
            "chunks_per_recipe": CHUNKS_PER_RECIPE,
            "status": info.status
        }
    
    def insert_recipe_chunks(
        self, 
        recipe: Dict[str, Any],
        chunk_embeddings: List[tuple],  # [(chunk_type, embedding), ...]
        parent_id: int
    ) -> int:
        """
        Tek bir tarifin chunk'larını veritabanına ekle
        
        Args:
            recipe: Tarif dictionary
            chunk_embeddings: [(chunk_type, embedding), ...] listesi
            parent_id: Parent (tarif) ID'si
        
        Returns:
            Eklenen chunk sayısı
        """
        points = []
        
        for chunk_idx, (chunk_type, embedding) in enumerate(chunk_embeddings):
            # Her chunk için benzersiz ID: parent_id * CHUNKS_PER_RECIPE + chunk_idx
            point_id = parent_id * CHUNKS_PER_RECIPE + chunk_idx
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    # Parent bilgileri (tam tarif)
                    "parent_id": parent_id,
                    "title": recipe.get("title", ""),
                    "url": recipe.get("url", ""),
                    "ingredients": recipe.get("ingredients", []),
                    "instructions": recipe.get("instructions", []),
                    
                    # Chunk bilgileri
                    "chunk_type": chunk_type,
                    "chunk_idx": chunk_idx,
                    
                    # Arama için ek alanlar
                    "ingredient_count": len(recipe.get("ingredients", [])),
                    "instruction_count": len(recipe.get("instructions", []))
                }
            )
            points.append(point)
        
        # Veritabanına ekle
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        return len(points)
    
    def insert_recipes_chunks(
        self, 
        recipes: List[Dict[str, Any]],
        all_chunk_embeddings: List[List[tuple]],  # [[recipe1_chunks], [recipe2_chunks], ...]
        start_parent_id: int = 0
    ) -> int:
        """
        Birden fazla tarifin chunk'larını veritabanına ekle
        
        Args:
            recipes: Tarif listesi
            all_chunk_embeddings: Her tarif için chunk embedding listesi
            start_parent_id: Başlangıç parent ID'si
        
        Returns:
            Eklenen toplam chunk sayısı
        """
        points = []
        
        for recipe_idx, (recipe, chunk_embeddings) in enumerate(zip(recipes, all_chunk_embeddings)):
            parent_id = start_parent_id + recipe_idx
            
            for chunk_idx, (chunk_type, embedding) in enumerate(chunk_embeddings):
                point_id = parent_id * CHUNKS_PER_RECIPE + chunk_idx
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "parent_id": parent_id,
                        "title": recipe.get("title", ""),
                        "url": recipe.get("url", ""),
                        "ingredients": recipe.get("ingredients", []),
                        "instructions": recipe.get("instructions", []),
                        "chunk_type": chunk_type,
                        "chunk_idx": chunk_idx,
                        "ingredient_count": len(recipe.get("ingredients", [])),
                        "instruction_count": len(recipe.get("instructions", []))
                    }
                )
                points.append(point)
        
        # Batch olarak ekle
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        return len(points)
    
    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        chunk_type_filter: Optional[str] = None,
        ingredient_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Vektör araması yap ve sonuçları parent'a göre grupla
        
        Args:
            query_vector: Sorgu vektörü
            top_k: Döndürülecek benzersiz tarif sayısı
            score_threshold: Minimum benzerlik skoru
            chunk_type_filter: Sadece belirli chunk türünde ara
            ingredient_filter: Belirli malzemeleri içeren tarifleri filtrele
        
        Returns:
            Bulunan tarifler listesi (parent bazlı, en iyi chunk skoru ile)
        """
        # Filtre oluştur
        must_conditions = []
        should_conditions = []
        
        if chunk_type_filter:
            must_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=chunk_type_filter)
                )
            )
        
        if ingredient_filter:
            for ing in ingredient_filter:
                should_conditions.append(
                    FieldCondition(
                        key="ingredients",
                        match=MatchText(text=ing)
                    )
                )
        
        query_filter = None
        if must_conditions or should_conditions:
            query_filter = Filter(
                must=must_conditions if must_conditions else None,
                should=should_conditions if should_conditions else None
            )
        
        # Daha fazla sonuç getir (parent'a göre gruplamak için)
        search_limit = top_k * CHUNKS_PER_RECIPE * 2
        
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=search_limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )
        
        # Sonuçları parent_id'ye göre grupla
        # Her parent için en iyi skoru tut
        parent_results = {}
        
        for result in response.points:
            parent_id = result.payload.get("parent_id")
            score = result.score
            chunk_type = result.payload.get("chunk_type")
            
            if parent_id not in parent_results or score > parent_results[parent_id]["score"]:
                parent_results[parent_id] = {
                    "id": parent_id,
                    "score": score,
                    "matched_chunk": chunk_type,
                    "title": result.payload.get("title", ""),
                    "url": result.payload.get("url", ""),
                    "ingredients": result.payload.get("ingredients", []),
                    "instructions": result.payload.get("instructions", [])
                }
        
        # Skora göre sırala ve top_k kadar döndür
        sorted_results = sorted(
            parent_results.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )[:top_k]
        
        return sorted_results
    
    def search_by_chunk_type(
        self,
        query_vector: List[float],
        chunk_type: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Belirli chunk türünde arama yap
        
        Args:
            query_vector: Sorgu vektörü
            chunk_type: "ingredients" veya "instructions"
            top_k: Döndürülecek sonuç sayısı
            score_threshold: Minimum benzerlik skoru
        
        Returns:
            Bulunan tarifler listesi
        """
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            chunk_type_filter=chunk_type
        )
    
    def get_recipe_by_parent_id(self, parent_id: int) -> Optional[Dict[str, Any]]:
        """Parent ID ile tarif getir"""
        # Parent'ın ilk chunk'ını getir
        point_id = parent_id * CHUNKS_PER_RECIPE
        
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id]
        )
        
        if results:
            point = results[0]
            return {
                "id": point.payload.get("parent_id"),
                "title": point.payload.get("title", ""),
                "url": point.payload.get("url", ""),
                "ingredients": point.payload.get("ingredients", []),
                "instructions": point.payload.get("instructions", [])
            }
        return None
    
    def delete_collection(self):
        """Collection sil"""
        if self.collection_exists():
            self.client.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Collection silindi: {COLLECTION_NAME}")
        else:
            print(f"ℹ️  Collection bulunamadı: {COLLECTION_NAME}")


# Singleton instance
_db_instance = None

def get_database() -> RecipeDatabase:
    """Database singleton instance döndür"""
    global _db_instance
    if _db_instance is None:
        _db_instance = RecipeDatabase()
    return _db_instance


if __name__ == "__main__":
    # Test
    db = get_database()
    
    print("\n📊 Collection Durumu:")
    info = db.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


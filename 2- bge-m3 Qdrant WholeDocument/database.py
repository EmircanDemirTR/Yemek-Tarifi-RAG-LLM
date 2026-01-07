"""
Qdrant Veritabanı Modülü
========================
Vektör veritabanı işlemleri
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
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchText
)
from config import (
    QDRANT_PATH, 
    COLLECTION_NAME, 
    EMBEDDING_DIM, 
    DISTANCE_METRIC,
    INDEX_BATCH_SIZE
)


class RecipeDatabase:
    """Qdrant vektör veritabanı işlemleri"""
    
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
        return {
            "exists": True,
            "points_count": info.points_count,
            "status": info.status
        }
    
    def insert_recipes(
        self, 
        recipes: List[Dict[str, Any]], 
        vectors: List[List[float]],
        start_id: int = 0
    ) -> int:
        """
        Tarifleri veritabanına ekle
        
        Args:
            recipes: Tarif listesi
            vectors: Embedding vektörleri
            start_id: Başlangıç ID'si
        
        Returns:
            Eklenen kayıt sayısı
        """
        points = []
        
        for i, (recipe, vector) in enumerate(zip(recipes, vectors)):
            point = PointStruct(
                id=start_id + i,
                vector=vector,
                payload={
                    "title": recipe.get("title", ""),
                    "url": recipe.get("url", ""),
                    "ingredients": recipe.get("ingredients", []),
                    "instructions": recipe.get("instructions", []),
                    # Arama için ek alanlar
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
        ingredient_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Vektör araması yap
        
        Args:
            query_vector: Sorgu vektörü
            top_k: Döndürülecek sonuç sayısı
            score_threshold: Minimum benzerlik skoru
            ingredient_filter: Belirli malzemeleri içeren tarifleri filtrele
        
        Returns:
            Bulunan tarifler listesi
        """
        # Filtre oluştur (isteğe bağlı)
        query_filter = None
        if ingredient_filter:
            # Malzeme filtreleme - herhangi biri içeren
            query_filter = Filter(
                should=[
                    FieldCondition(
                        key="ingredients",
                        match=MatchText(text=ing)
                    )
                    for ing in ingredient_filter
                ]
            )
        
        # Yeni Qdrant API - query_points kullan
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter
        )
        
        # Sonuçları düzenle
        formatted_results = []
        for result in response.points:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "title": result.payload.get("title", ""),
                "url": result.payload.get("url", ""),
                "ingredients": result.payload.get("ingredients", []),
                "instructions": result.payload.get("instructions", [])
            })
        
        return formatted_results
    
    def get_recipe_by_id(self, recipe_id: int) -> Optional[Dict[str, Any]]:
        """ID ile tarif getir"""
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[recipe_id]
        )
        
        if results:
            point = results[0]
            return {
                "id": point.id,
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


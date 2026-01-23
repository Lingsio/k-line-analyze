
import asyncio
import sys
import os
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.auto_analyzer import AutoAnalyzer
from app.services.feature_extractor import FeatureExtractor
from app.services.similarity_search import SimilaritySearchEngine
from app.config import settings

async def main():
    print("Initializing AutoAnalyzer components...")
    
    try:
        analyzer = AutoAnalyzer()
        print("AutoAnalyzer initialized.")
    except Exception as e:
        print(f"Failed to init AutoAnalyzer: {e}")
        return

    print(f"Index loaded: {analyzer.search_engine.is_index_loaded()}")
    print(f"Total vectors: {analyzer.search_engine.get_total_vectors()}")
    
    if not analyzer.search_engine.is_index_loaded():
        print("Index NOT loaded. Looking for index files...")
        print(f"Index dir: {settings.FAISS_INDEX_DIR}")
        if settings.FAISS_INDEX_DIR.exists():
            print("Directory exists.")
            print("Files:", os.listdir(settings.FAISS_INDEX_DIR))
        else:
            print("Directory does NOT exist.")
        return

    # Try to extract feature for a mock dataframe
    print("\nTesting Feature Extraction...")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=20)
    df = pd.DataFrame({
        'open': np.linspace(100, 110, 20),
        'high': np.linspace(101, 111, 20),
        'low': np.linspace(99, 109, 20),
        'close': np.linspace(100.5, 110.5, 20),
        'volume': np.random.randint(1000, 5000, 20)
    }, index=dates)
    
    try:
        embedding = analyzer.feature_extractor.extract_from_dataframe(df)
        print(f"Embedding extracted successfully. Shape: {embedding.shape}")
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Search
    print("\nTesting Search...")
    try:
        results = analyzer.search_engine.search(embedding, top_k=5)
        print(f"Found {len(results)} results:")
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['symbol']} ({res['market']}) - Score: {res['similarity_score']:.4f}, Date: {res.get('start_date')} to {res.get('end_date')}")
            
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

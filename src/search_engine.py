import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.prompt_preprocess import MemeSearchPreprocessor

class MemeSearchEngine:
    def __init__(self, data_path: str, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Инициализация поискового движка.
        
        Args:
            data_path: путь к файлу с датасетом и эмбеддингами (.parquet)
            model_name: название модели для эмбеддингов
        """
        self.df = pd.read_parquet(data_path)
        self.model = SentenceTransformer(model_name)
        self.meme_embeddings = np.array(self.df['embedding'].tolist())
        self.preprocessor = MemeSearchPreprocessor()
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> pd.DataFrame:
        """
        Гибридный поиск с поддержкой количества и минимальной схожести.
        
        Args:
            query: текстовый запрос
            top_k: максимальное количество возвращаемых результатов
            min_similarity: минимальный порог схожести (0.0-1.0)
                
        Returns:
            DataFrame с результатами, удовлетворяющими обоим условиям
        """
        query_processed = self.preprocessor.preprocess(query, for_search=True)
        query_embedding = self.model.encode([query_processed], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.meme_embeddings)[0]

        if min_similarity > 0:
            mask = similarities >= min_similarity
            eligible_indices = np.where(mask)[0]
        else:
            eligible_indices = np.arange(len(similarities))
        
        if len(eligible_indices) == 0:
            return pd.DataFrame()

        sorted_indices = eligible_indices[np.argsort(similarities[eligible_indices])[::-1]]
        k = min(int(top_k), len(sorted_indices))
        top_indices = sorted_indices[:k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx].copy()
            row['score'] = float(similarities[idx])
            results.append(row)
        
        return pd.DataFrame(results)
    
    def get_image_bytes(self, idx: int) -> bytes:
        """Получить байты изображения по индексу в DataFrame."""
        return self.df.iloc[idx]['local_path']

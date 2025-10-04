import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

class FeatureEngineer:
    def __init__(self, movies_df):
        self.movies_df = movies_df.copy()
    
    def create_genre_features(self):
        """Create genre-based features as sparse matrix"""
        print("Creating genre features...")
        
        # Get all unique genres
        all_genres = set()
        for genres in self.movies_df['genres']:
            if genres != '(no genres listed)':
                all_genres.update(genres.split('|'))
        
        genre_list = sorted(list(all_genres))
        
        # Create genre matrix
        genre_matrix = np.zeros((len(self.movies_df), len(genre_list)))
        
        for i, genres in enumerate(self.movies_df['genres']):
            if genres != '(no genres listed)':
                for genre in genres.split('|'):
                    if genre in genre_list:
                        genre_matrix[i, genre_list.index(genre)] = 1
        
        print(f"Created {len(genre_list)} genre features")
        return csr_matrix(genre_matrix), np.array(genre_list)
    
    def create_content_features(self):
        """Create content-based features using TF-IDF on movie titles"""
        print("Creating content features...")
        
        # Use movie titles for content features
        titles = self.movies_df['title'].fillna('').astype(str)
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = tfidf.fit_transform(titles)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        return tfidf_matrix
    
    def create_hybrid_features(self):
        """Create hybrid features combining genres and content"""
        genre_features, genre_names = self.create_genre_features()
        content_features = self.create_content_features()
        
        # Combine features (horizontal stacking)
        from scipy.sparse import hstack
        hybrid_features = hstack([genre_features, content_features])
        
        return hybrid_features, genre_names
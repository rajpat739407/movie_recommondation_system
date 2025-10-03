from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

class FeatureEngineer:
    def __init__(self, movies_df):
        self.movies = movies_df
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')  # Reduced features
        self.mlb = MultiLabelBinarizer()
    
    def create_genre_features(self):
        """Create genre-based features"""
        print("Creating genre features...")
        
        # One-hot encode genres
        genre_features = self.mlb.fit_transform(self.movies['genres_list'])
        genre_feature_names = self.mlb.classes_
        
        print(f"Created {len(genre_feature_names)} genre features")
        return genre_features, genre_feature_names
    
    def create_content_features(self):
        """Create content-based features from movie titles and genres"""
        print("Creating content features...")
        
        # Combine title and genres for content analysis
        self.movies['content'] = self.movies['title'] + ' ' + self.movies['genres'].str.replace('|', ' ')
        
        # Create TF-IDF features with reduced dimensions
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['content'])
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix
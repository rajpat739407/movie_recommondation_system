import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, movies_df, ratings_df, min_ratings=5):
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        self.min_ratings = min_ratings
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.user_item_matrix = None
        
    def preprocess_movies(self):
        """Preprocess movies data"""
        print("Preprocessing movies data...")
        
        # Handle missing values
        self.movies_df['title'] = self.movies_df['title'].fillna('Unknown')
        self.movies_df['genres'] = self.movies_df['genres'].fillna('(no genres listed)')
        
        # Extract genres
        all_genres = set()
        for genres in self.movies_df['genres']:
            if genres != '(no genres listed)':
                all_genres.update(genres.split('|'))
        
        print(f"Found {len(all_genres)} unique genres: {all_genres}")
        return self.movies_df
    
    def preprocess_ratings(self):
        """Preprocess ratings data"""
        print("Preprocessing ratings data...")
        
        # Filter out users and movies with too few ratings
        user_rating_counts = self.ratings_df['userId'].value_counts()
        movie_rating_counts = self.ratings_df['movieId'].value_counts()
        
        filtered_users = user_rating_counts[user_rating_counts >= self.min_ratings].index
        filtered_movies = movie_rating_counts[movie_rating_counts >= self.min_ratings].index
        
        self.ratings_df = self.ratings_df[
            self.ratings_df['userId'].isin(filtered_users) & 
            self.ratings_df['movieId'].isin(filtered_movies)
        ]
        
        print(f"After filtering: {len(self.ratings_df)} ratings")
        return self.ratings_df
    
    def create_user_item_matrix(self):
        """Create sparse user-item matrix"""
        print("Creating sparse user-item matrix...")
        
        # Encode user and movie IDs
        encoded_users = self.user_encoder.fit_transform(self.ratings_df['userId'])
        encoded_movies = self.movie_encoder.fit_transform(self.ratings_df['movieId'])
        
        # Create sparse matrix
        self.user_item_matrix = csr_matrix(
            (self.ratings_df['rating'], (encoded_users, encoded_movies)),
            shape=(len(self.user_encoder.classes_), len(self.movie_encoder.classes_))
        )
        
        print(f"Sparse user-item matrix shape: {self.user_item_matrix.shape}")
        
        # Calculate sparsity
        sparsity = (1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100
        print(f"Sparsity: {sparsity:.2f}%")
        print(f"Number of ratings: {self.ratings_df['rating'].count()}")
        print(f"Matrix size: {self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]}")
        
        return self.user_item_matrix
    
    def get_preprocessed_data(self):
        """Get all preprocessed data"""
        movies_processed = self.preprocess_movies()
        ratings_processed = self.preprocess_ratings()
        user_item_matrix = self.create_user_item_matrix()
        
        return movies_processed, ratings_processed, user_item_matrix
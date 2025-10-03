import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

class DataPreprocessor:
    def __init__(self, movies, ratings):
        self.movies = movies.copy()
        self.ratings = ratings.copy()
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
    
    def preprocess_movies(self):
        """Preprocess movies data"""
        print("Preprocessing movies data...")
        
        # Extract year from title
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')
        self.movies['year'] = self.movies['year'].fillna(0).astype(int)
        
        # Split genres into lists
        self.movies['genres_list'] = self.movies['genres'].str.split('|')
        
        # Get all unique genres
        all_genres = set()
        for genres in self.movies['genres_list']:
            all_genres.update(genres)
        
        print(f"Found {len(all_genres)} unique genres: {all_genres}")
        return self.movies
    
    def preprocess_ratings(self):
        """Preprocess ratings data"""
        print("Preprocessing ratings data...")
        
        # Filter out users with too few ratings
        user_rating_counts = self.ratings['userId'].value_counts()
        active_users = user_rating_counts[user_rating_counts >= 5].index
        self.ratings = self.ratings[self.ratings['userId'].isin(active_users)]
        
        # Filter out movies with too few ratings
        movie_rating_counts = self.ratings['movieId'].value_counts()
        popular_movies = movie_rating_counts[movie_rating_counts >= 10].index
        self.ratings = self.ratings[self.ratings['movieId'].isin(popular_movies)]
        
        print(f"After filtering: {len(self.ratings)} ratings")
        return self.ratings
    
    def create_sparse_user_item_matrix(self):
        """Create sparse user-item matrix using scipy sparse matrix"""
        print("Creating sparse user-item matrix...")
        
        # Encode user and movie IDs to continuous indices
        self.ratings['user_idx'] = self.user_encoder.fit_transform(self.ratings['userId'])
        self.ratings['movie_idx'] = self.movie_encoder.fit_transform(self.ratings['movieId'])
        
        # Create sparse matrix directly (more memory efficient)
        n_users = len(self.ratings['user_idx'].unique())
        n_movies = len(self.ratings['movie_idx'].unique())
        
        # Use scipy sparse matrix to avoid memory issues
        sparse_matrix = csr_matrix(
            (self.ratings['rating'], 
             (self.ratings['user_idx'], self.ratings['movie_idx'])),
            shape=(n_users, n_movies)
        )
        
        print(f"Sparse user-item matrix shape: {sparse_matrix.shape}")
        
        # Calculate sparsity
        matrix_size = sparse_matrix.shape[0] * sparse_matrix.shape[1]
        num_ratings = sparse_matrix.nnz
        sparsity = 100 * (1 - (num_ratings / matrix_size))
        print(f"Sparsity: {sparsity:.2f}%")
        print(f"Number of ratings: {num_ratings}")
        print(f"Matrix size: {matrix_size}")
        
        return sparse_matrix
    
    def get_preprocessed_data(self):
        """Run all preprocessing steps"""
        movies_processed = self.preprocess_movies()
        ratings_processed = self.preprocess_ratings()
        user_item_matrix = self.create_sparse_user_item_matrix()  # Updated method name
        
        return movies_processed, ratings_processed, user_item_matrix
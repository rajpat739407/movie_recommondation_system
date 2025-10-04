import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
import pickle
from sklearn.neighbors import NearestNeighbors
import os

class MovieRecommendationEngine:
    def __init__(self, movies_df, user_encoder, movie_encoder):
        self.movies_df = movies_df
        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder
        self.user_item_matrix = None
        self.knn_model = None
        self.svd_model = None
        self.factorized_matrix = None
        
    def load_trained_models(self, models_path='../models'):
        """Load pre-trained models"""
        print("Loading trained models...")
        
        try:
            # Get current directory and construct paths
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Load user-item matrix
            user_item_path = os.path.join(current_dir, 'data', 'user_item_matrix_sparse.npz')
            self.user_item_matrix = load_npz(user_item_path)
            
            # Load KNN model
            knn_path = os.path.join(models_path, 'knn_model.pkl')
            with open(knn_path, 'rb') as f:
                self.knn_model = pickle.load(f)
                
            # Load SVD model
            svd_path = os.path.join(models_path, 'svd_model.pkl')
            with open(svd_path, 'rb') as f:
                self.svd_model = pickle.load(f)
                
            # Load factorized matrix
            factorized_path = os.path.join(models_path, 'factorized_matrix.npy')
            self.factorized_matrix = np.load(factorized_path)
                
            print("All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_user_recommendations(self, user_id, method='hybrid', n_recommendations=10):
        """Get recommendations for a specific user"""
        
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
        except:
            return f"User ID {user_id} not found"
        
        if method == 'user_based':
            return self._knn_recommendations(user_idx, n_recommendations)
        elif method == 'item_based':
            return self._svd_recommendations(user_idx, n_recommendations)
        elif method == 'hybrid':
            return self._hybrid_recommendations(user_idx, n_recommendations)
        else:
            return "Invalid method"
    
    def _knn_recommendations(self, user_idx, n_recommendations):
        """KNN-based collaborative filtering recommendations"""
        if self.knn_model is None:
            return "KNN model not loaded"
        
        # Find similar users
        distances, indices = self.knn_model.kneighbors(
            self.user_item_matrix[user_idx], 
            n_neighbors=20
        )
        
        similar_users = indices.flatten()[1:]  # Exclude the user itself
        
        # Get movies rated by similar users but not by target user
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Average ratings from similar users
        similar_users_ratings = self.user_item_matrix[similar_users].mean(axis=0).A.flatten()
        
        # Mask out movies already rated
        similar_users_ratings[user_ratings > 0] = 0
        
        # Get top recommendations
        top_movie_indices = np.argsort(similar_users_ratings)[::-1][:n_recommendations]
        
        recommendations = []
        for movie_idx in top_movie_indices:
            movie_id = self.movie_encoder.inverse_transform([movie_idx])[0]
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                score = similar_users_ratings[movie_idx]
                recommendations.append({
                    'title': movie_info['title'].values[0],
                    'genres': movie_info['genres'].values[0],
                    'score': float(score)
                })
        
        return recommendations
    
    def _svd_recommendations(self, user_idx, n_recommendations):
        """SVD-based recommendations"""
        if self.svd_model is None or self.factorized_matrix is None:
            return "SVD model not loaded"
        
        # Get user's factor vector
        user_factors = self.factorized_matrix[user_idx]
        
        # Reconstruct user's ratings (approximate)
        predicted_ratings = user_factors @ self.svd_model.components_
        
        # Get actual ratings to mask already rated items
        actual_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        predicted_ratings[actual_ratings > 0] = 0  # Mask rated items
        
        # Get top recommendations
        top_movie_indices = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        
        recommendations = []
        for movie_idx in top_movie_indices:
            movie_id = self.movie_encoder.inverse_transform([movie_idx])[0]
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                score = predicted_ratings[movie_idx]
                recommendations.append({
                    'title': movie_info['title'].values[0],
                    'genres': movie_info['genres'].values[0],
                    'score': float(score)
                })
        
        return recommendations
    
    def _hybrid_recommendations(self, user_idx, n_recommendations):
        """Hybrid recommendations combining KNN and SVD"""
        knn_recs = self._knn_recommendations(user_idx, n_recommendations)
        svd_recs = self._svd_recommendations(user_idx, n_recommendations)
        
        if isinstance(knn_recs, str) or isinstance(svd_recs, str):
            return "Error in loading models"
        
        # Combine and remove duplicates
        all_recs = knn_recs + svd_recs
        unique_recs = []
        seen_titles = set()
        
        for rec in all_recs:
            if rec['title'] not in seen_titles:
                unique_recs.append(rec)
                seen_titles.add(rec['title'])
        
        return unique_recs[:n_recommendations]
    
    def get_movie_similarity(self, movie_id, n_similar=5):
        """Find similar movies using SVD latent factors"""
        try:
            # Check if movie exists in our dataset
            if movie_id not in self.movie_encoder.classes_:
                return f"Movie ID {movie_id} not found in dataset"
            
            movie_idx = self.movie_encoder.transform([movie_id])[0]
            
            # Check if movie_idx is within bounds of SVD components
            if movie_idx >= self.svd_model.components_.shape[0]:
                return f"Movie index {movie_idx} is out of bounds for the model"
            
            # Use SVD components for movie similarity
            movie_factors = self.svd_model.components_[movie_idx]
            
            # Calculate cosine similarity with all other movies
            similarities = []
            n_movies = min(len(self.movie_encoder.classes_), self.svd_model.components_.shape[0])
            
            for other_idx in range(n_movies):
                if other_idx != movie_idx:
                    other_factors = self.svd_model.components_[other_idx]
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(movie_factors, other_factors)
                    norm_a = np.linalg.norm(movie_factors)
                    norm_b = np.linalg.norm(other_factors)
                    
                    if norm_a > 0 and norm_b > 0:
                        similarity = dot_product / (norm_a * norm_b)
                    else:
                        similarity = 0
                    
                    similarities.append((other_idx, similarity))
            
            # Get top similar movies
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar_indices = [idx for idx, sim in similarities[:n_similar]]
            
            similar_movies_list = []
            for idx in top_similar_indices:
                try:
                    similar_movie_id = self.movie_encoder.inverse_transform([idx])[0]
                    movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id]
                    if not movie_info.empty:
                        similarity_score = next(sim for i, sim in similarities if i == idx)
                        similar_movies_list.append({
                            'title': movie_info['title'].values[0],
                            'genres': movie_info['genres'].values[0],
                            'similarity_score': float(similarity_score)
                        })
                except Exception as e:
                    continue  # Skip if there's an error with this movie
            
            if not similar_movies_list:
                return "No similar movies found"
            
            return similar_movies_list
            
        except Exception as e:
            return f"Error finding similar movies: {str(e)}"
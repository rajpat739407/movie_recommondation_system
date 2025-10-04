import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import pickle
import time

class CollaborativeFiltering:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.knn_model = None
        
    def approximate_user_similarity(self, n_neighbors=50, metric='cosine'):
        """Use KNN for approximate similarity (memory efficient)"""
        print("Calculating approximate user-user similarity using KNN...")
        start_time = time.time()
        
        # Use KNN for approximate similarity search
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors, 
            metric=metric, 
            algorithm='brute', 
            n_jobs=-1
        )
        
        self.knn_model.fit(self.user_item_matrix)
        
        print(f"KNN model fitted in {time.time() - start_time:.2f} seconds")
        return self.knn_model
    
    def item_based_similarity_optimized(self, sample_size=5000):
        """Calculate item similarity on a sample of items (memory efficient)"""
        print("Calculating item-item similarity on sample...")
        start_time = time.time()
        
        # Work with item-user matrix (transpose)
        item_user_matrix = self.user_item_matrix.T
        
        # Sample items to reduce computation
        n_items = item_user_matrix.shape[0]
        if n_items > sample_size:
            # Random sample of items
            sample_indices = np.random.choice(n_items, size=sample_size, replace=False)
            item_user_sample = item_user_matrix[sample_indices]
        else:
            item_user_sample = item_user_matrix
            sample_indices = np.arange(n_items)
        
        # Calculate cosine similarity on sample
        from sklearn.metrics.pairwise import cosine_similarity
        item_similarity_sample = cosine_similarity(item_user_sample, dense_output=False)
        
        print(f"Item similarity calculated on {len(sample_indices)} items in {time.time() - start_time:.2f} seconds")
        return item_similarity_sample, sample_indices
    
    def matrix_factorization(self, n_components=100):
        """Apply SVD for matrix factorization"""
        print("Applying Matrix Factorization with SVD...")
        start_time = time.time()
        
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.factorized_matrix = self.svd_model.fit_transform(self.user_item_matrix)
        
        print(f"Explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.4f}")
        print(f"Factorized matrix shape: {self.factorized_matrix.shape}")
        print(f"SVD completed in {time.time() - start_time:.2f} seconds")
        
        return self.factorized_matrix
    
    def get_user_recommendations_knn(self, user_idx, n_recommendations=10, n_neighbors=20):
        """Get recommendations using KNN approach"""
        if self.knn_model is None:
            self.approximate_user_similarity(n_neighbors=n_neighbors)
        
        # Find similar users
        distances, indices = self.knn_model.kneighbors(
            self.user_item_matrix[user_idx], 
            n_neighbors=n_neighbors
        )
        
        similar_users = indices.flatten()[1:]  # Exclude the user itself
        similarities = 1 - distances.flatten()[1:]  # Convert distances to similarities
        
        # Get movies rated by similar users but not by target user
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Average ratings from similar users
        similar_users_ratings = self.user_item_matrix[similar_users].mean(axis=0).A.flatten()
        
        # Mask out movies already rated
        similar_users_ratings[user_ratings > 0] = 0
        
        # Get top recommendations
        top_movie_indices = np.argsort(similar_users_ratings)[::-1][:n_recommendations]
        
        return top_movie_indices, similar_users_ratings[top_movie_indices]
    
    def get_svd_recommendations(self, user_idx, n_recommendations=10):
        """Get recommendations using SVD matrix factorization"""
        if self.svd_model is None:
            self.matrix_factorization()
        
        # Get user's factor vector
        user_factors = self.factorized_matrix[user_idx]
        
        # Reconstruct user's ratings (approximate)
        predicted_ratings = user_factors @ self.svd_model.components_
        
        # Get actual ratings to mask already rated items
        actual_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        predicted_ratings[actual_ratings > 0] = 0  # Mask rated items
        
        # Get top recommendations
        top_movie_indices = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        
        return top_movie_indices, predicted_ratings[top_movie_indices]
    
    def get_hybrid_recommendations(self, user_idx, n_recommendations=10):
        """Combine KNN and SVD recommendations"""
        # Get recommendations from both methods
        knn_indices, knn_scores = self.get_user_recommendations_knn(user_idx, n_recommendations * 2)
        svd_indices, svd_scores = self.get_svd_recommendations(user_idx, n_recommendations * 2)
        
        # Combine and deduplicate
        all_recommendations = {}
        
        for idx, score in zip(knn_indices, knn_scores):
            all_recommendations[idx] = all_recommendations.get(idx, 0) + score * 0.5
            
        for idx, score in zip(svd_indices, svd_scores):
            all_recommendations[idx] = all_recommendations.get(idx, 0) + score * 0.5
        
        # Sort by combined score
        sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Return top n
        top_indices = [idx for idx, score in sorted_recommendations[:n_recommendations]]
        top_scores = [score for idx, score in sorted_recommendations[:n_recommendations]]
        
        return top_indices, top_scores
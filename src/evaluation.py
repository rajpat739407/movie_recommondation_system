import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import pandas as pd

class Evaluator:
    def __init__(self, ratings_df, user_item_matrix):
        self.ratings_df = ratings_df
        self.user_item_matrix = user_item_matrix
        
    def train_test_split_ratings(self, test_size=0.2, random_state=42):
        """Split ratings into train and test sets"""
        print("Splitting data into train/test sets...")
        
        # Create train/test mask
        train_data, test_data = train_test_split(
            self.ratings_df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.ratings_df['userId']  # Maintain user distribution
        )
        
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        return train_data, test_data
    
    def create_matrices_from_data(self, train_data, test_data):
        """Create train and test matrices from split data"""
        # Create user and movie encoders based on train data
        train_users = train_data['user_idx'].unique()
        train_movies = train_data['movie_idx'].unique()
        
        # Create train matrix
        train_matrix = csr_matrix(
            (train_data['rating'], 
             (train_data['user_idx'], train_data['movie_idx'])),
            shape=self.user_item_matrix.shape
        )
        
        # Create test matrix (only include users and movies from train)
        test_matrix = csr_matrix(
            (test_data['rating'], 
             (test_data['user_idx'], test_data['movie_idx'])),
            shape=self.user_item_matrix.shape
        )
        
        return train_matrix, test_matrix
    
    def calculate_rmse(self, actual_ratings, predicted_ratings):
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    
    def calculate_precision_recall(self, actual_binary, predicted_binary, k=10):
        """Calculate Precision@K and Recall@K"""
        precision = precision_score(actual_binary, predicted_binary)
        recall = recall_score(actual_binary, predicted_binary)
        return precision, recall
    
    def coverage(self, recommended_items, all_items):
        """Calculate coverage - percentage of items that can be recommended"""
        unique_recommended = len(set(recommended_items))
        return unique_recommended / len(all_items)
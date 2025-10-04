import pickle
import numpy as np
from scipy.sparse import save_npz, load_npz
import os

def save_model(model, filepath):
    """Save model to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Load model from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_sparse_matrix(matrix, filepath):
    """Save sparse matrix to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_npz(filepath, matrix)

def load_sparse_matrix(filepath):
    """Load sparse matrix from file"""
    return load_npz(filepath)

def get_movie_title(movie_id, movies_df):
    """Get movie title from movie ID"""
    movie_info = movies_df[movies_df['movieId'] == movie_id]
    if len(movie_info) > 0:
        return movie_info['title'].iloc[0]
    return f"Movie {movie_id}"

def validate_user_id(user_id, user_encoder):
    """Validate if user ID exists in the system"""
    return user_id in user_encoder.classes_

def validate_movie_id(movie_id, movie_encoder):
    """Validate if movie ID exists in the system"""
    return movie_id in movie_encoder.classes_
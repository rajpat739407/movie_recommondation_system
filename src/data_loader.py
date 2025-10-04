import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path='data/ml-latest-small'):
        self.data_path = data_path
    
    def load_movies(self):
        return pd.read_csv(os.path.join(self.data_path, 'movies.csv'))
    
    def load_ratings(self):
        return pd.read_csv(os.path.join(self.data_path, 'ratings.csv'))
    
    def load_links(self):
        return pd.read_csv(os.path.join(self.data_path, 'links.csv'))
    
    def load_all_data(self):
        movies = self.load_movies()
        ratings = self.load_ratings()
        links = self.load_links()
        return movies, ratings, links

# Usage example
if __name__ == "__main__":
    loader = DataLoader()
    movies, ratings, links = loader.load_all_data()
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings")
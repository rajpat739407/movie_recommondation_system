import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from recommendation_engine import MovieRecommendationEngine

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
        border-left: 4px solid #FF4B4B;
    }
    .similarity-score {
        color: #FF4B4B;
        font-weight: bold;
    }
    .recommendation-score {
        color: #00D4AA;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data_and_models():
    """Load all data and models with caching"""
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct absolute paths
        data_dir = os.path.join(current_dir, 'data')
        models_dir = os.path.join(current_dir, 'models')
        
        # Load data
        movies_path = os.path.join(data_dir, 'movies_processed.csv')
        movies_processed = pd.read_csv(movies_path)
        
        # Load encoders
        user_encoder_path = os.path.join(data_dir, 'user_encoder.pkl')
        movie_encoder_path = os.path.join(data_dir, 'movie_encoder.pkl')
        
        with open(user_encoder_path, 'rb') as f:
            user_encoder = pickle.load(f)
        
        with open(movie_encoder_path, 'rb') as f:
            movie_encoder = pickle.load(f)
        
        # Initialize and load models
        engine = MovieRecommendationEngine(movies_processed, user_encoder, movie_encoder)
        success = engine.load_trained_models(models_dir)
        
        if success:
            st.success("✅ Models loaded successfully!")
            return engine, movies_processed
        else:
            st.error("❌ Failed to load models")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner('Loading movie data and AI models...'):
        engine, movies_df = load_data_and_models()
    
    if engine is None:
        st.error("""
        ❌ Failed to load the recommendation system. 
        
        **Please ensure:**
        1. All model files exist in the `models/` directory
        2. All data files exist in the `data/` directory
        3. You have run the model training notebooks first
        
        Run `notebooks/03_model_training.ipynb` to train the models if you haven't already.
        """)
        
        # Show file structure for debugging
        st.subheader("📁 Current File Structure")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Directory:**")
            data_dir = os.path.join(current_dir, 'data')
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    st.write(f"📄 {file}")
            else:
                st.write("❌ data/ directory not found")
        
        with col2:
            st.write("**Models Directory:**")
            models_dir = os.path.join(current_dir, 'models')
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    st.write(f"📄 {file}")
            else:
                st.write("❌ models/ directory not found")
        
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a feature:",
        ["🏠 Home", "🔍 Search Movies", "🎯 Get Recommendations", "📊 Analytics", "ℹ️ About"]
    )
    
    # Main content based on selection
    if app_mode == "🏠 Home":
        show_homepage(movies_df)
    elif app_mode == "🔍 Search Movies":
        show_search_movies(movies_df, engine)
    elif app_mode == "🎯 Get Recommendations":
        show_recommendations(movies_df, engine)
    elif app_mode == "📊 Analytics":
        show_analytics(movies_df)
    elif app_mode == "ℹ️ About":
        show_about()

def show_homepage(movies_df):
    """Homepage with overview and statistics"""
    st.header("Welcome to Your Personal Movie Guide! 🎥")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Movies", f"{len(movies_df):,}")
    
    with col2:
        st.metric("Unique Genres", movies_df['genres'].str.split('|').explode().nunique())
    
    with col3:
        try:
            ratings_df = pd.read_csv('data/ratings_processed.csv')
            st.metric("Total Ratings", f"{len(ratings_df):,}")
        except:
            st.metric("Total Ratings", "N/A")
    
    # Popular movies
    st.subheader("🎭 Most Popular Genres")
    genre_counts = movies_df['genres'].str.split('|').explode().value_counts().head(10)
    fig = px.bar(
        x=genre_counts.values, 
        y=genre_counts.index,
        orientation='h',
        title="Top 10 Movie Genres",
        labels={'x': 'Number of Movies', 'y': 'Genre'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample movies
    st.subheader("🎬 Sample Movies from Collection")
    sample_movies = movies_df.head(10)
    for _, movie in sample_movies.iterrows():
        with st.container():
            year_display = f" | 📅 {int(movie['year'])}" if 'year' in movie and movie['year'] > 0 else ""
            st.markdown(f"""
            <div class="movie-card">
                <strong>{movie['title']}</strong><br>
                🎭 {movie['genres']}{year_display}
            </div>
            """, unsafe_allow_html=True)

def show_search_movies(movies_df, engine):
    """Movie search functionality"""
    st.header("🔍 Search Movies")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input("Search for movies by title:", placeholder="e.g., The Matrix, Inception...")
    
    with col2:
        genre_filter = st.selectbox("Filter by genre:", ["All Genres"] + sorted(movies_df['genres'].str.split('|').explode().unique()))
    
    # Search results
    if search_query:
        search_results = movies_df[movies_df['title'].str.contains(search_query, case=False, na=False)]
    else:
        search_results = movies_df
    
    # Apply genre filter
    if genre_filter != "All Genres":
        search_results = search_results[search_results['genres'].str.contains(genre_filter, na=False)]
    
    st.subheader(f"Found {len(search_results)} movies")
    
    # Display results
    for _, movie in search_results.head(20).iterrows():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            year_display = f" | 📅 {int(movie['year'])}" if 'year' in movie and movie['year'] > 0 else ""
            st.write(f"**{movie['title']}**")
            st.write(f"🎭 {movie['genres']}{year_display}")
        
        with col2:
            if st.button("Find Similar", key=f"similar_{movie['movieId']}"):
                with st.spinner('Finding similar movies...'):
                    similar_movies = engine.get_movie_similarity(movie['movieId'], n_similar=3)
                    if isinstance(similar_movies, list):
                        st.session_state[f'similar_to_{movie["movieId"]}'] = similar_movies
                    else:
                        st.error(similar_movies)
        
        # Show similar movies if available
        if f'similar_to_{movie["movieId"]}' in st.session_state:
            st.info(f"🎬 Movies similar to **{movie['title']}**:")
            for similar in st.session_state[f'similar_to_{movie["movieId"]}']:
                st.write(f"   • {similar['title']} (Similarity: <span class='similarity-score'>{similar['similarity_score']:.3f}</span>)", unsafe_allow_html=True)
        
        st.divider()

def show_recommendations(movies_df, engine):
    """Movie recommendations for users"""
    st.header("🎯 Get Personalized Recommendations")
    
    try:
        ratings_df = pd.read_csv('data/ratings_processed.csv')
        available_users = ratings_df['userId'].value_counts().head(50).index.tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_id = st.selectbox("Select User ID:", available_users)
        
        with col2:
            method = st.selectbox(
                "Recommendation Method:",
                ["hybrid", "user_based", "item_based"],
                format_func=lambda x: {
                    "hybrid": "🤝 Hybrid (Best)",
                    "user_based": "👥 User-Based", 
                    "item_based": "🎬 Item-Based"
                }[x]
            )
        
        with col3:
            n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("🎯 Get Recommendations", type="primary"):
            with st.spinner('Finding the perfect movies for you...'):
                recommendations = engine.get_user_recommendations(
                    user_id, 
                    method=method, 
                    n_recommendations=n_recommendations
                )
            
            if isinstance(recommendations, list) and recommendations:
                st.success(f"🎉 Found {len(recommendations)} recommendations for User {user_id}")
                
                # Display recommendations in a nice grid
                cols = st.columns(2)
                for i, rec in enumerate(recommendations):
                    with cols[i % 2]:
                        with st.container():
                            st.markdown(f"""
                            <div class="movie-card">
                                <h4>#{i+1} {rec['title']}</h4>
                                <p>🎭 {rec['genres']}</p>
                                <p>⭐ Recommendation Score: <span class='recommendation-score'>{rec['score']:.3f}</span></p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Could not generate recommendations: {recommendations}")
        
        # Show user's rating history
        st.subheader("📊 User's Rating History")
        user_ratings = ratings_df[ratings_df['userId'] == user_id].merge(
            movies_df, on='movieId'
        ).sort_values('rating', ascending=False)
        
        if not user_ratings.empty:
            st.write(f"User {user_id} has rated {len(user_ratings)} movies")
            
            # Display top rated movies
            top_rated = user_ratings.head(10)
            for _, rating in top_rated.iterrows():
                st.write(f"⭐ {rating['rating']}/5.0 - **{rating['title']}**")
        else:
            st.info("No rating history available for this user.")
            
    except Exception as e:
        st.error(f"Error loading ratings data: {e}")

def show_analytics(movies_df):
    """Data analytics and insights"""
    st.header("📊 System Analytics & Insights")
    
    try:
        ratings_df = pd.read_csv('data/ratings_processed.csv')
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", ratings_df['userId'].nunique())
        with col2:
            st.metric("Total Movies", ratings_df['movieId'].nunique())
        with col3:
            st.metric("Total Ratings", len(ratings_df))
        with col4:
            avg_rating = ratings_df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")
        
        # Rating distribution
        st.subheader("📈 Rating Distribution")
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Distribution of Movie Ratings",
            labels={'x': 'Rating', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")

def show_about():
    """About page"""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🎬 Movie Recommendation System
    
    This is a comprehensive movie recommendation system built using machine learning techniques 
    to provide personalized movie suggestions based on user preferences and movie characteristics.
    
    ### 🚀 Features:
    - **🔍 Search Movies**: Find movies by title and genre
    - **🎯 Personalized Recommendations**: Get movie suggestions based on collaborative filtering
    - **🤝 Hybrid Recommendations**: Combine multiple algorithms for better accuracy
    - **📊 Analytics**: Explore data insights and statistics
    - **🎬 Similar Movies**: Discover movies similar to your favorites
    
    ### 🛠️ Technologies Used:
    - **Python** with Scikit-learn, Pandas, NumPy
    - **Collaborative Filtering** (User-based, Item-based)
    - **Matrix Factorization** (SVD)
    - **K-Nearest Neighbors** for similarity search
    - **Streamlit** for web interface
    
    Developed with ❤️ for movie enthusiasts!
    """)

if __name__ == "__main__":
    main()

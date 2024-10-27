import streamlit as st
import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
import numpy as np
from surprise.model_selection import cross_validate

#Load Model
import pickle
# Load data back from the file
with open('66130701714recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Function to get movie recommendations
def get_recommendations(user_id, n_factors=20, top_n=10):
    # Ensure user ID exists in the ratings data
    if user_id not in movie_ratings['user_id'].values:
        return pd.DataFrame(columns=['movieId', 'title', 'estimated_rating'])
    
    # Predict ratings for all movies for this user and get the top ones
    movies['estimated_rating'] = movies['movieId'].apply(lambda x: svd_model.predict(user_id, x, n_factors).est)
    recommended_movies = movies.sort_values(by='estimated_rating', ascending=False).head(top_n)
    return recommended_movies[['title', 'estimated_rating']]
    
# Load the MovieLens dataset (ml-latest-small)
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Merge movies and ratings data
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Streamlit App
st.title('Top 10 Reccomendation Movie')

st.write("Provide your User ID to get top 10 movie recommendations.")

# User input: User ID
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

# Factor selection
factor_options = st.slider("Select SVD Factor Count (higher = more factors):", min_value=10, max_value=100, value=20, step=10)

# Button to generate recommendations
if st.button("Get Recommendations"):
    if user_id:
        recommendations = get_recommendations(user_id, factor_options)
        st.write("Top 10 Movie Recommendations for User ID:", user_id)
        
        # Display recommendations
        for i, (movie, rating) in enumerate(recommendations, 1):
            st.write(f"{i}. {movie} - Predicted Rating: {rating:.2f}")
    else:
        st.warning("Please enter a valid User ID.")

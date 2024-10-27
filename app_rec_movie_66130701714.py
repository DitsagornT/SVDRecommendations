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

# Load the pre-trained SVD model, movie ratings, and movies data
with open('66130701714recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)
 
# Streamlit app title and description
st.title("Movie Recommendation System")
st.write("Get the top 10 movie recommendations based on your user ID.")
 
# User input for user ID
user_id = st.number_input("Enter your User ID:", min_value=1, step=1)
 
if st.button("Get Recommendations"):
    # Get movies rated by the user
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
 
    # Predict ratings for unrated movies
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
 
    # Sort predictions by estimated rating in descending order
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
 
    # Get top 10 movie recommendations
    top_recommendations = sorted_predictions[:10]
 
    # Display top recommendations
    st.write(f"### Top 10 movie recommendations for User {user_id}:")
    for recommendation in top_recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")
 

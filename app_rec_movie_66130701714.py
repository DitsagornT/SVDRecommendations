import pickle
import streamlit as st
from surprise import SVD

# Load data back from the file
with open('66130701714recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Streamlit application
st.title("Movie Recommendation System")

# User input for user ID
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

# Check if the user ID exists in the ratings data
if user_id not in movie_ratings['userId'].values:
    st.write("User ID not found in the dataset. Please enter a valid User ID.")
else:
    # Get rated movies for the user
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values

    if len(rated_user_movies) == 0:
        st.write("No rated movies found for this User ID.")
    else:
        # Find unrated movies
        unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

        # Predict ratings for unrated movies
        pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]

        # Sort predictions by estimated rating in descending order
        sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)

        # Get top 10 movie recommendations
        top_recommendations = sorted_predictions[:10]

        # Display top recommendations
        st.subheader(f"Top 10 Movie Recommendations for User {user_id}:")
        for recommendation in top_recommendations:
            movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
            st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")

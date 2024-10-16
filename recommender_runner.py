import joblib
import pandas as pd
from tfidf import BookRecommender  # Adjust the import as needed

def get_recommendations(book_title, top_n=5):
    # Initialize your recommender
    recommender = BookRecommender()  # Make sure the path is correct
    recommender.preprocess_data()      # Preprocess any necessary data if needed
    recommender.compute_similarity()    # Compute the similarity matrix

    # Get recommended books for the given title
    recommended_books = recommender.recommend_books(book_title, top_n=top_n)  
    return recommended_books[['title', 'description', 'book_id']]

def main():
    # Specify a book title for recommendations
    title_to_recommend = "Smoke"  # Replace with a valid title from your dataset
    recommended_books = get_recommendations(title_to_recommend, top_n=5)

    # Print the recommended books
    print("Recommended Books:")
    print(recommended_books)

if __name__ == "__main__":
    main()

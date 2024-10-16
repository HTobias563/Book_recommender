import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse
import joblib

class BookRecommender:
    def __init__(self, data_path='data/books_tfidf.csv'):  # Updated default path
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)  # Load dataset from the specified path
        self.vectorizer = None
        self.cosine_sim = None
        self.df_original = None

    def preprocess_data(self):
        # Text columns to process with TF-IDF
        text_cols = ['description', 'title']
        num_cols = ['book_id']  # Adjust if you have more numerical features
        
        # Fill missing values in numerical columns with 0
        self.df[num_cols] = self.df[num_cols].fillna(0)
        
        # Combine text columns into one for TF-IDF processing
        self.df['features_combined'] = self.df[text_cols].fillna('').agg(' '.join, axis=1)

        # Scale numerical features between 0 and 1 for similarity calculation
        scaler = MinMaxScaler()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])

        # Keep a copy of the original data to display recommendations later
        self.df_original = self.df.copy()

    def compute_similarity(self):
        # TF-IDF vectorization on combined text features
        self.vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = self.vectorizer.fit_transform(self.df['features_combined'])

        # Convert numerical data to a sparse matrix
        numerical_matrix = sparse.csr_matrix(self.df[['book_id']].values)

        # Combine the TF-IDF matrix with scaled numerical features
        combined_matrix = sparse.hstack([tfidf_matrix, numerical_matrix])

        # Calculate cosine similarity on the combined matrix
        self.cosine_sim = cosine_similarity(combined_matrix, combined_matrix)

    def recommend_books(self, title, top_n=5):
        if title not in self.df['title'].values:
            print(f"The book '{title}' is not in the dataset.")
            return pd.DataFrame()

        # Get the index of the book by title
        idx = self.df[self.df['title'] == title].index[0]

        # Get similarity scores for all books
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # Skip the first entry (it's the book itself)

        # Get indices of the most similar books
        book_indices = [i[0] for i in sim_scores]

        # Return the most similar books
        return self.df_original.iloc[book_indices]

    def save_vectorizer(self, filepath='tfidf_vectorizer.pkl'):
        # Save the vectorizer model for reuse
        joblib.dump(self.vectorizer, filepath)

# Example of how to use this class
if __name__ == "__main__":
    recommender = BookRecommender()  # Path to 'data/books_tfidf.csv'
    recommender.preprocess_data()
    recommender.compute_similarity()
    
    # Replace "Smoke" with a valid book title from your dataset
    recommended_books = recommender.recommend_books("Smoke", top_n=5)  
    print(recommended_books[['title', 'description', 'book_id']])
    
    # Save the TF-IDF vectorizer model
    recommender.save_vectorizer('tfidf_vectorizer.pkl')

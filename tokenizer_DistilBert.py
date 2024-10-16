import pandas as pd
import torch
from transformers import DistilBertTokenizer
import numpy as np

# Step 1: Load the CSV files
file_path_train = 'data/filtered_books_fina1.csv'
file_path_test = 'data/filtered_books_final_test.csv'
df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)

# Step 2: Check for null values and remove rows with missing values if necessary
print("Missing values in train set:", df_train.isnull().sum())
print("Missing values in test set:", df_test.isnull().sum())

df_train.dropna(subset=['description', 'title', 'similar_book'], inplace=True)
df_test.dropna(subset=['description', 'title', 'similar_book'], inplace=True)

# Step 3: Initialize the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize descriptions and titles
df_train['description_tokens'] = df_train['description'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=512))
df_train['title_tokens'] = df_train['title'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=512))

df_test['description_tokens'] = df_test['description'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=512))
df_test['title_tokens'] = df_test['title'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=512))

# Step 4: Extract the target variable (similar_books)
y_train = df_train['similar_book'].values  # Target variable for training
y_test = df_test['similar_book'].values    # Target variable for testing

# Step 5: Create input variables (X) for training data
X_train_description = torch.tensor(df_train['description_tokens'].tolist())
X_train_title = torch.tensor(df_train['title_tokens'].tolist())
X_train_book_id = torch.tensor(df_train['book_id'].values)

# Create input variables (X) for test data
X_test_description = torch.tensor(df_test['description_tokens'].tolist())
X_test_title = torch.tensor(df_test['title_tokens'].tolist())
X_test_book_id = torch.tensor(df_test['book_id'].values)

# Step 6: Reshape X_author and X_book_id tensors
X_train_book_id = X_train_book_id.unsqueeze(1)
X_test_book_id = X_test_book_id.unsqueeze(1)

# Step 7: Combine the input variables into a single tensor for training and test data
X_train_combined = torch.cat((X_train_description, X_train_title, X_train_book_id), dim=1)
X_test_combined = torch.cat((X_test_description, X_test_title, X_test_book_id), dim=1)

# Step 8: Save the tokenized data as NumPy arrays
np.savez('data/tokenized_books.npz', 
         X_train=X_train_combined.numpy(), 
         X_test=X_test_combined.numpy(), 
         y_train=y_train, 
         y_test=y_test)

# Output the dimensions of the datasets
print(f'Train size: {len(y_train)}, Test size: {len(y_test)}')

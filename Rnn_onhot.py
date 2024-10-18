import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step: Load the tokenized data
data = np.load('data/tokenized_books.npz')

# Extract the data
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Output the dimensions of the data
print(f'Train size: {len(y_train)}, Test size: {len(y_test)}')

# Reshape to (batch_size, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Add a time step dimension
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])  # Add a time step dimension

# Step: Create a mapping from old labels to new continuous labels
y_combined = np.concatenate([y_train, y_test])
y_combined, _ = np.unique(y_combined, return_inverse=True)
book_mapping = dict(zip(y_combined, list(range(len(y_combined)))))

# Map y_train and y_test to continuous integer labels
y_train = np.array([book_mapping[y] for y in y_train])
y_test = np.array([book_mapping[y] for y in y_test])

# Convert labels to one-hot encoding
num_classes = len(book_mapping)  # Number of unique books (classes)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Build the Model
model = keras.Sequential()

# LSTM expects input in shape (batch_size, timesteps, features)
input_dim = X_train.shape[2]  # Dimension of the combined input features
model.add(layers.Input(shape=(1, input_dim)))  # Input layer for LSTM

# Dropout layer for regularization
model.add(layers.Dropout(0.2))  # After the first LSTM layer

# LSTM Layers
model.add(layers.LSTM(128, return_sequences=True))  # Return sequences for stacked LSTM
model.add(layers.LSTM(64))  # Second LSTM layer

# Fully connected layers
model.add(layers.Dense(64, activation='relu'))  # First dense layer
model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer (num_classes)

# Summary of the model
model.summary()

# Compile the model with categorical crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using the one-hot encoded labels
batch_size = 32
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test))

# Save the model in .keras format with a new name to reflect one-hot encoding
model.save('my_model_one_hot.keras')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Output the evaluation results
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plotting the training and validation accuracy
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

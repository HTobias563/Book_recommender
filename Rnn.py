import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

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

print('y train map : ', y_test)
y_combined = np.concatenate([y_train, y_test])
y_combined, _ = np.unique(np.concatenate([y_train, y_test]), return_inverse=True)
book_mapping = dict(zip(y_combined,list(range(0,len(y_combined)))))
for i, y_i in enumerate(y_train):
    y_train[i] = book_mapping[y_i]

for i, y_i in enumerate(y_test):
    y_test[i] = book_mapping[y_i]

print(y_train)
print(y_test)


# Step 1: Create a mapping from old labels to new continuous labels
unique_labels,_ = np.unique(y_train, return_inverse=True)
print('y train map : ', _)
#_, y_test_mapped = np.unique(y_test, return_inverse=True)

# Hyperparameters
batch_size = 32
num_classes = len(unique_labels)  # Number of unique books (classes)

input_dim = X_train.shape[2]  # Dimension of the combined input features

# Build the Model
model = keras.Sequential()

# LSTM expects input in shape (batch_size, timesteps, features)
model.add(layers.Input(shape=(1, input_dim)))  # Input layer for LSTM

#dropout 
model.add(layers.Dropout(0.2))  # After the first LSTM layer

# LSTM Layers
model.add(layers.LSTM(128, return_sequences=True))  # Return sequences for stacked LSTM
model.add(layers.LSTM(64))  # Second LSTM layer

# Fully connected layers
model.add(layers.Dense(64, activation='relu'))  # First dense layer
model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer (num_classes)

# Summary of the model
model.summary()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using the mapped labels
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test))

# Save the model in .keras format
model.save('my_model.keras')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Output the evaluation results
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')


import matplotlib.pyplot as plt

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


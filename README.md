# Sentiment Analysis Project Documentation

## Project Overview

This project involves building a sentiment analysis model using TensorFlow and Keras. The model is designed to classify text reviews into different sentiment categories. The project includes data preprocessing, model training, evaluation, and saving the trained model for future use.

## Dependencies

To run this project, you need the following libraries:

- NumPy
- Pandas
- TensorFlow
- Matplotlib
- Seaborn
- Scikit-learn

You can install these libraries using pip:

```bash
pip install numpy pandas tensorflow matplotlib seaborn scikit-learn
```

## Data

The dataset used in this project is `Sentiment_Data.csv`. It contains text reviews and their corresponding sentiment labels. The dataset is loaded and preprocessed using Pandas.

## Code Explanation

### Import Libraries

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load and Explore Dataset

```python
# Load dataset
df = pd.read_csv('Sentiment_Data.csv')

# Display dataset information
df.info()

# Drop the 'Rating' column
df = df.drop(['Rating'], axis=1)

# Plot the distribution of sentiments
plt.bar(df['Sentiment'].value_counts().index, df['Sentiment'].value_counts().values)
```

### Visualize Sentiment Distribution

```python
# Count plot for sentiment distribution
ax = sns.countplot(x='Sentiment', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.xlabel("Sentiment", fontweight='bold')
plt.ylabel("Frequency", fontweight='bold')
plt.show()
```

### Preprocess Text Data

```python
# Parameters for text preprocessing
vocab_size = 1000
embedding_dim = 16
max_length = 80
trunc_type = 'post'
padding_type = 'post'
OOV_tok = '<OOV>'

# Extract sentences and labels
sentences = df['Review']
labels = df['Sentiment']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
sentences_train, sentences_test, labels_train, labels_test = train_test_split(
    sentences, labels, train_size=0.8, shuffle=False
)
sentences_val = sentences_test

# Tokenize sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=OOV_tok)
tokenizer.fit_on_texts(sentences_train)
word_index = tokenizer.word_index

# Convert sentences to sequences and pad them
train_sequences = tokenizer.texts_to_sequences(sentences_train)
sentences_train = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

test_sequences = tokenizer.texts_to_sequences(sentences_test)
sentences_test = pad_sequences(test_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

# Tokenize labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
labels_train = np.array(label_tokenizer.texts_to_sequences(labels_train))
labels_test = np.array(label_tokenizer.texts_to_sequences(labels_test))
```

### Build and Train Model

```python
# Define the model
model_sentiment_analysis = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')
])

# Custom callback to stop training based on accuracy
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') is not None and logs.get('val_accuracy') is not None:
            if logs.get('accuracy') > 0.85 and logs.get('val_accuracy') > 0.85:
                self.model.stop_training = True

callbacks = myCallback()

# Compile the model
model_sentiment_analysis.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

# Train the model
model_sentiment_analysis.fit(
    sentences_train,
    labels_train,
    epochs=10,
    validation_data=(sentences_test, labels_test),
    # callbacks=callbacks
)
```

### Save and Load Model

```python
# Save the trained model
model_sentiment_analysis.save('sentiment_analysis.h5')

# Load the saved model
model = tf.keras.models.load_model('sentiment_analysis.h5')
```

### Evaluate Model

```python
# Prepare validation data
sequences_val = tokenizer.texts_to_sequences(sentences_val)
sentences_val = pad_sequences(sequences_val, maxlen=max_length, truncating=trunc_type, padding=padding_type)

# Predict sentiment for validation data
predictions = model.predict(sentences_val)
predicted_class = np.argmax(predictions, axis=1)
```

## Repository Structure

```
.
├── Sentiment_Data.csv
├── sentiment_analysis.py
└── README.md
```

- `Sentiment_Data.csv`: Dataset containing text reviews and sentiment labels.
- `Sentiment_Analysis.ipynb`: Main script for loading data, preprocessing, building, training, and evaluating the model.
- `README.md`: Documentation of the project.


## Conclusion

This project demonstrates how to build a sentiment analysis model using TensorFlow and Keras. The model is trained on a dataset of text reviews and can classify the sentiment of new reviews with high accuracy. The trained model is saved for future use, allowing easy deployment in applications.

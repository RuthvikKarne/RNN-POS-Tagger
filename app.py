import os
import re
import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
MODEL_PATH = 'emission.keras'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_WORDS = 5000
MAX_LEN = 50

# POS tag categories
POS_TAGS = ['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X']


def clean_text(text):
    """Clean and normalize text input."""
    text = str(text).lower()
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def train_model():
    """Train the POS tagger model."""
    st.info('Training model...')
    print('Loading and preprocessing dataset...')
    
    # Check dataset exists
    if not os.path.exists('emission_probs.csv'):
        st.error("Error: 'emission_probs.csv' not found! Please place it in the same directory.")
        return False
    
    # Load and preprocess dataset
    df = pd.read_csv('emission_probs.csv', encoding='latin-1')
    
    # FIXED: Proper column selection (word + POS columns)
    required_cols = ['word'] + POS_TAGS
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
        return False
    
    df = df[required_cols].copy()
    
    import ast
    def extract_tuple(text):
        try:
            t = ast.literal_eval(text)
            return str(t[0]), str(t[1])
        except:
            return str(text), 'X'
            
    # Extract actual word and target tag
    extracted = df['word'].apply(extract_tuple)
    df['word'] = extracted.apply(lambda x: x[0]).apply(clean_text)
    df['target_tag'] = extracted.apply(lambda x: x[1])
    
    df = df[df['word'] != '']  # Remove empty strings
    
    # Prepare tokenizer (char_level=True is critical so the RNN processes characters rather than treating the whole word as 1 token)
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<oov>', char_level=True)
    tokenizer.fit_on_texts(df['word'])
    
    # Save tokenizer
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Convert words to sequences
    sequences = tokenizer.texts_to_sequences(df['word'])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    
    # Labels: one-hot encode the actual extracted tags rather than using the CSV float probabilities
    y = np.zeros((len(df), len(POS_TAGS)))
    for i, tag in enumerate(df['target_tag']):
        if tag in POS_TAGS:
            y[i, POS_TAGS.index(tag)] = 1.0
        else:
            y[i, POS_TAGS.index('X')] = 1.0
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        padded, y, test_size=0.2, random_state=42
    )
    
    from tensorflow.keras.layers import Bidirectional, LSTM, Dropout
    
    # IMPROVED: Model architecture (Bidirectional LSTM, proper Multiclass classification)
    model = Sequential([
        Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(POS_TAGS), activation="softmax")  # Softmax for single-class classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Architecture:")
    model.summary()
    
    # Train model
    from tensorflow.keras.callbacks import EarlyStopping
    with st.spinner('Training...'):
        history = model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=64,
            validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
            verbose=1
        )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Predictions (argmax for categorical)
    predictions_probs = model.predict(X_test)
    predictions_idx = np.argmax(predictions_probs, axis=1)
    
    # Convert true labels from one-hot to index
    y_test_idx = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test_idx, predictions_idx, target_names=POS_TAGS, labels=range(len(POS_TAGS))))
    
    # Save model
    model.save(MODEL_PATH)
    st.success("Training complete! Model saved.")
    return True


def predict_word(word):
    """
    Predict POS tags for a single word.
    
    Args:
        word: Input word string
        
    Returns:
        dict: Dictionary mapping POS tags to confidence scores
    """
    # Load model and tokenizer if not already loaded
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at '{MODEL_PATH}'. Please train the model first.")
        return None
    
    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"Tokenizer not found at '{TOKENIZER_PATH}'. Please train the model first.")
        return None
    
    # Load components
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Preprocess input
    cleaned = clean_text(word)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0]  # Shape: (12,)
    predicted_idx = np.argmax(prediction)
    
    # Format results
    results = {}
    for i, tag in enumerate(POS_TAGS):
        results[tag] = {
            'probability': float(prediction[i]),
            'predicted': (i == predicted_idx)
        }
    
    return results


def predict_word_streamlit():
    """Streamlit UI wrapper for predict_word()."""
    st.subheader("POS Tag Prediction")
    
    word = st.text_input("Enter a word:", "")
    
    if st.button("Predict") and word:
        results = predict_word(word)
        
        if results:
            st.write(f"### Results for: '{word}'")
            
            # Find the predicted tag
            predicted_tag = None
            probability = 0.0
            for tag, info in results.items():
                if info['predicted']:
                    predicted_tag = tag
                    probability = info['probability']
                    break
            
            if predicted_tag:
                st.success(f"Predicted POS tag: **{predicted_tag}**")
                st.info(f"Confidence: **{probability * 100:.2f}%**")
            else:
                st.warning("Could not determine POS tag.")


def main():
    """Main Streamlit app."""
    st.title("POS Tagger with RNN")
    
    menu = st.sidebar.selectbox(
        "Menu",
        ["Train Model", "Predict", "About"]
    )
    
    if menu == "Train Model":
        if st.button("Start Training"):
            train_model()
            
    elif menu == "Predict":
        predict_word_streamlit()
        
    else:
        st.markdown("""
        ## About
        This is a Part-of-Speech (POS) tagger using a SimpleRNN.
        
        **Supported POS Tags:**
        - `.` — Punctuation
        - `ADJ` — Adjective
        - `ADP` — Adposition
        - `ADV` — Adverb
        - `CONJ` — Conjunction
        - `DET` — Determiner
        - `NOUN` — Noun
        - `NUM` — Number
        - `PRON` — Pronoun
        - `PRT` — Particle
        - `VERB` — Verb
        - `X` — Other
        """)


if __name__ == "__main__":
    main()
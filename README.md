# RNN-POS-Tagger

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

A character-level Bidirectional LSTM neural network built with TensorFlow and Streamlit that predicts Part-of-Speech (POS) tags for isolated words with real-time confidence scoring.

## 🚀 Features

- **Character-Level RNN:** Utilizes a Bidirectional LSTM network to predict POS tags based on character sequences, helping capture morphological features.
- **Interactive Web App:** Built with Streamlit for a clean, user-friendly interface.
- **Train in the Browser:** Easily train the model from scratch directly within the Streamlit app.
- **Real-Time Prediction:** Get confidence scores and POS tag predictions instantly.
- **Support for 12 POS Tags:** Classifies words into detailed categories like Nouns, Verbs, Adjectives, Adverbs, and more.

## 🧠 Model Architecture

The neural network is designed for sequence classification at the character level:
1. **Embedding Layer** (Converts character indices into dense vectors)
2. **Bidirectional LSTM** (Captures forward and backward character context)
3. **Dropout Layers** (Prevents overfitting)
4. **Dense Layers with ReLU & Softmax** (For final multiclass classification)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RuthvikKarne/RNN-POS-Tagger.git
   cd RNN-POS-Tagger
   ```

2. **Install dependencies:**
   Make sure you have Python installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Provide the dataset:**
   Ensure `emission_probs.csv` is present in the root directory. This dataset is required to train the model.

## 💻 Usage

Run the Streamlit application locally:

```bash
streamlit run app.py
```

### Steps in the App:
1. Navigate to the **Train Model** menu in the sidebar and click **Start Training**.
2. Once the model is trained and saved (`emission.keras` & `tokenizer.pkl`), switch to the **Predict** menu.
3. Enter any word to get its Part-of-Speech tag and confidence score.

## 🏷️ Supported POS Tags

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

## 📁 Repository Structure

```
RNN-POS-Tagger/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── emission_probs.csv         # Dataset for training (Add this!)
├── POS_NER_with_RNNs.ipynb    # Experimental Jupyter Notebook
└── README.md                  # Project documentation
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/RuthvikKarne/RNN-POS-Tagger/issues).

# Khmer Semantic Similarity Detection using RNN-based Representation Learning

This project focuses on detecting semantic similarity between Khmer words using character-level representation learning and Gated Recurrent Units (GRU/LSTM/RNN). It addresses specific challenges in Khmer NLP like stacked orthography, non-concatenative morphology, and segmentation dependencies.

## 🚀 Project Overview

The core idea is to map discrete, variable-length Khmer word sequences into a continuous, fixed-size vector space where semantically similar words are geometrically close to each other.

### Key Features
*   **Character-level Processing**: Handles Khmer Unicode characters directly, including consonants, vowels, and diacritics.
*   **Siamese Architecture**: Uses a shared encoder for word pairs to ensure consistent representation.
*   **Multi-Model Support**: Implements SimpleRNN, LSTM, and GRU variants.
*   **Two-Stage Learning**:
    1.  **Stage 1 (Unsupervised)**: Autoencoder-based reconstruction to learn the Khmer word manifold.
    2.  **Stage 2 (Supervised)**: Siamese classification to distinguish synonyms from non-synonyms.

---

## 🛠 Step-by-Step Guide

Follow these steps to understand and run the project:

### 1. Data Loading & Exploration
*   Load the dataset from `khmer_semantic_similarity_dataset_chars.csv`.
*   Explore the word pairs and their similarity labels (1 for similar, 0 for dissimilar).
*   The project uses character sequences (e.g., `['ក', 'ន', '្', 'ល', 'ែ', 'ង']`) as input.

### 2. Preprocessing & Vocabulary Building
*   Extract all unique characters from the dataset to build a **Character Vocabulary**.
*   **Encoding**: Convert characters into unique integers (e.g., 'ក' → 1, 'ន' → 2).
*   **Padding**: Pad all sequences to a fixed `max_len` (typically 100) to allow batch processing.

### 3. Model Architecture Construction
*   **Shared Embedding Layer**: Converts character indices into dense feature vectors.
*   **RNN Encoder**: A shared LSTM/GRU layer that processes the sequence and outputs a "Context Vector" representing the word.
*   **Similarity calculation**: Combine the two context vectors (using specialized layers like Dot product or Manhattan distance).
*   **Classification Head**: Dense layers with a Sigmoid activation to output a similarity probability.

### 4. Training the Model
*   Split the data into training and testing sets (80/20 split).
*   Compile the model with `binary_crossentropy` loss and the `adam` optimizer.
*   Train for multiple epochs (e.g., 20 epochs) while monitoring `val_accuracy`.

### 5. Evaluation & Testing
*   Evaluate the model on the test set to check accuracy and loss.
*   Generate a `classification_report` to see Precision, Recall, and F1-score.
*   Use the `is_similar(word1, word2)` function to test new word pairs.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `Pre-trained.ipynb` | Notebook for pre-training character embeddings and representation learning. |
| `similarity (1).ipynb` | Main notebook for similarity detection and Siamese model training. |
| `Section2_Update.md` | Detailed theoretical framework and mathematical mechanics. |
| `*.csv` | Datasets containing Khmer words, character tokens, and similarity pairs. |
| `*.pkl` | Pickled preprocessing data (char2idx, max_len) for deployment. |

---

## ⚙️ Setup & Requirements

1.  **Environment**: Python 3.8+, TensorFlow 2.x, Pandas, Scikit-learn.
2.  **Data**: Ensure all `.csv` files are in the project root or adjust paths in the notebooks.
3.  **Run**: Open `similarity (1).ipynb` in Jupyter or Kaggle and run all cells to train and test the model.

---

## 📖 Theoretical Background

The project is grounded in **Manifold Learning**. The Autoencoder stage compresses high-dimensional sparse character space into a dense manifold, while the Siamese stage learns the separation hyperplane for synonyms.

For a deeper dive into the math, refer to `Section2_Update.md`.
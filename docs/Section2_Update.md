
# 2. Problem Statement and Theoretical Framework

## 2.1 Formal Problem Definition

The task of Semantic Similarity Detection can be cast as a supervised learning problem within the domain of representation learning. We define the problem formally through set theory and functional mapping, providing a step-by-step breakdown of the underlying mechanics.

### 2.1.1 Input Space and Vocabulary
Let $\Sigma$ represent the universal character set defined by the Khmer Unicode Block (U+1780 to U+17FF). This set includes:
*   **Consonants** ($\mathcal{C}$): 35 base characters (e.g., ក, ខ, គ).
*   **Dependent Vowels** ($\mathcal{V}$): Symbols modifying the inherent vowel (e.g., ា, ិ, ី).
*   **Diacritics** ($\mathcal{D}$): Signs altering pronunciation (e.g., ៉, ៊).
*   **Layout Controls:** Specifically the Zero-Width Joiner (ZWJ) and the sign Coeng (U+17D2), denoted as $\dagger$.

We define a "word" $w$ as a finite sequence of characters from this alphabet:
$$ w = (x_1, x_2, \dots, x_T) \quad \text{where} \quad x_t \in \Sigma $$
The space of all legitimate Khmer words is denoted as $\mathcal{W} \subset \Sigma^*$. High-dimensional discrete space.

### 2.1.2 The Mathematical Goal
Our goal is to map this discrete, variable-length sequence $w$ into a continuous, fixed-size vector space $\mathbb{R}^d$ (e.g., a list of 256 numbers) where geometry represents meaning.

We designate this mapping function as the **Encoder**, $E_\phi$:
$$ E_\phi: \mathcal{W} \rightarrow \mathbb{R}^d $$

If the model works correctly, the vector for "Computer" ($v_1$) and "Laptop" ($v_2$) should be very close to each other in this space, while "Computer" and "Banana" ($v_3$) should be far apart. Mathematically, we want to minimize the Euclidean distance $||v_1 - v_2||$ for synonyms.

## 2.2 Mathematical Mechanics of the Model (How it Works)

To understand *how* the model converts a sequence of characters into a concept, we must look at the equations of the **Gated Recurrent Unit (GRU)**. This is the "brain" of our system.

### 2.2.1 The Embedding Step
First, each character $x_t$ (e.g., 'ក') is effectively a one-hot vector of size $|\Sigma|$. We project this into a dense vector $e_t$ using an embedding matrix $W_e$:
$$ e_t = W_e \cdot x_t $$
This step converts the discrete symbol into a set of learnable features.

### 2.2.2 The Recurrent Step (The "Memory")
The model processes the word one character at a time. At each step $t$, it maintains a "hidden state" vector $h_t$, which acts as its short-term memory of what it has read so far.

The transition from $h_{t-1}$ (previous state) to $h_t$ (current state) is governed by the GRU equations. These equations allow the model to decide **what to keep** and **what to forget**.

1.  **Update Gate ($z_t$):**
    *   *Equation:* $z_t = \sigma(W_z \cdot [h_{t-1}, e_t])$
    *   *Meaning:* This gate is a number between 0 and 1. It decides "How much of the *new* information should I overwrite the old memory with?" If $z_t \approx 0$, the model ignores the new character. If $z_t \approx 1$, it updates its belief.

2.  **Reset Gate ($r_t$):**
    *   *Equation:* $r_t = \sigma(W_r \cdot [h_{t-1}, e_t])$
    *   *Meaning:* This decides "How much of the past is irrelevant?" For example, when reading a new syllable in a compound word, the reset gate might signal the model to focus less on the previous syllable.

3.  **Candidate Heritage ($\tilde{h}_t$):**
    *   *Equation:* $\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, e_t])$
    *   *Meaning:* This calculates the *potential* new memory, combining the current character $e_t$ with the relevant parts of the past ($r_t \odot h_{t-1}$).

4.  **Final State Update:**
    *   *Equation:* $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$
    *   *Meaning:* The new state is a weighted average of the old state and the candidate state.

After processing the entire word sequence of length $T$, the final hidden state $h_T$ represents the **Context Vector**—a dense summary of the entire word's structure and meaning.
$$ v_{word} = h_T $$

## 2.3 The Optimization Objective

We train this mathematical structure using two different Loss Functions ($\mathcal{L}$) corresponding to our two stages.

### 2.3.1 Stage 1: Reconstruction Loss (Autoencoder)
In the first stage, the model tries to recreate the input word. If input is $x$, and output is $\hat{x}$.
We use **Categorical Cross-Entropy**:
$$ \mathcal{L}_{recon} = - \sum_{t=1}^{T} \sum_{c \in \Sigma} y_{t,c} \log(\hat{y}_{t,c}) $$
*   Where $y_{t,c}$ is 1 if character $c$ is actually at position $t$, and 0 otherwise.
*   **Intuition:** This forces the vector $h_T$ to store *all* necessary information to reconstruct the word (spelling, morphology). If the vector "forgets" an infix, the reconstruction fails, and the loss increases.

### 2.3.2 Stage 2: Similarity Loss (Binary Classification)
In the second stage, we take two vectors $u$ and $v$ (from word A and word B) and combine them into a feature vector $f$.
$$ f = [u; v; |u-v|; u \cdot v] $$
We pass $f$ into a classifier to get a probability $\hat{y}$. We minimize **Binary Cross-Entropy**:
$$ \mathcal{L}_{sim} = - [y \log(\hat{y}) + (1-y) \log(1-\hat{y})] $$
*   **Intuition:** If $y=1$ (Similar), we want $\hat{y}$ to be close to 1. If $\hat{y}$ is 0.1, the term $\log(\hat{y})$ becomes a large negative number, creating a large loss (penalty) that forces the model to adjust its weights.

## 2.4 Theoretical Challenges in Khmer NLP

To understand why this formalization is difficult to solve, one must understand the specific linguistic and orthographic hurdles inherent to the Khmer language.

### 2.4.1 The "Stacked" Orthography (The Coeng Problem)
Unlike linear alphabets (Latin, Cyrillic), Khmer script utilizes a two-dimensional positioning system. A consonant can carry a "subscript" consonant (using the Coeng marker), a vowel (which can be visually placed left, right, above, or below), and a diacritic.

**Mathematical implication for Sequence Modeling:**
Recurrent Neural Networks (RNNs) assume a temporal sequence $t=1, \dots, T$. In Khmer, the "visual" order does not match the "logical" Unicode order.
*   *Visual Order:* Vowel `e` (េ) appears *before* the consonant it modifies.
*   *Logical Order:* Consonant appears *before* Vowel `e` (េ) in memory.
*   *Ambiguity:* Some legacy fonts/encodings allow users to type in visual order, creating "ghost" words that look correct but are encoded wrongly ($w_{visual} \neq w_{logical}$). The model must be robust enough to learn the canonical representation despite these inconsistencies in the wild data.

### 2.4.2 Non-Concatenative Morphology
English morphology is largely concatenative (suffixing): *walk* $\rightarrow$ *walk-ing*.
Khmer morphology often involves **infixation**, where a modifier is inserted *inside* the root word.
*   *Root:* **K-E-T** (កើត - to allow/happen)
*   *Derived:* **K-MN-E-T** (កំណើត - origin)
The infix `-mn-` splits the root characters. A standard bag-of-words model or a Convolutional Neural Network (CNN) with a small kernel size might fail to link **KET** and **KMNET** because the contiguous character patterns are disrupted. This necessitates an RNN architecture with a long-term memory (like LSTM or GRU) that can maintain the "state" of the root consonant $K$ across the infix to link it with the final $T$.

### 2.4.3 The Segmentation-Semantic Cycle
Deep learning models for semantics usually rely on pre-segmented words (Word2Vec). However, in Khmer, segmentation often *depends* on semantics, creating a chicken-and-egg problem.
*   Consider the string: **"tak-te"**.
    *   Interpretation A: "Water" (Tuek) + "Tea" (Te) = *Tea Water*.
    *   Interpretation B: "To decorate" (Tok-teng).
Without understanding the semantic context of the sentence, one cannot determine the correct word boundary. By training a character-level model on full sentence reconstruction (Stage 1), we aim to implicitly teach the model these boundary probabilities before it attempts the explicit similarity task.

## 2.5 Limits of Existing Solutions

### 2.5.1 Statistical Approaches (TF-IDF / Co-occurrence)
Traditional vectors based on Term Frequency-Inverse Document Frequency (TF-IDF) or Matrix Factorization fail in this context because of the **Sparsity Problem**. In an agglutinative language, the vocabulary size $V$ grows almost linearly with corpus size due to the formation of compound words. A TF-IDF matrix becomes overwhelmingly sparse ($|\mathcal{W}| > 100,000$), making similarity computations unreliable for rare words.

### 2.5.2 Zero-Shot Multilingual Models (mBERT, XLM-R)
While Multilingual BERT (mBERT) includes Khmer in its training data, it is severely undertrained compared to English. Studies have shown that mBERT often resorts to "transliteration" matching rather than deep semantic understanding for low-resource languages. Furthermore, mBERT uses a WordPiece tokenizer. For Khmer, this tokenizer often shatters words into nonsensical byte fragments, destroying the morphological structure described in Section 2.2.2.

## 2.6 The Proposed Framework: Distributional Representation

We adopt the **Distributional Hypothesis**: *Linguistic items with similar distributions have similar meanings.*
However, we adapt this hypothesis for the character level. instead of defining "distribution" as neighboring *words*, we define it as neighboring *characters* and the predictively capability of the sequence.

If the sequence of characters $E(w_1)$ can be reconstructed by the same decoder weights as the sequence $E(w_2)$, or if they map to the same manifold in $\mathbb{R}^d$, they share structural similarity. Our Two-Stage approach is theoretically grounded in **Manifold Learning**:
1.  **Stage 1 (Unsupervised):** Learn the manifold of valid Khmer words. The Autoencoder compresses the high-dimensional sparse character space into a low-dimensional dense manifold.
2.  **Stage 2 (Supervised):** Learn a separation hyperplane on this manifold that distinguishes "synonyms" from "non-synonyms".

This theoretical separation allows us to tackle the data scarcity problem (Section 2.3.1) by ensuring that the dense manifold is well-formed using abundant unlabeled data before we attempt to learn the separation boundary.

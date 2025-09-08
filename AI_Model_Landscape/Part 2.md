#Author : [Tarriq Ferrose Khan](www.linkedin.com/in/tarriq-ferrose-khan-ba527080) 

<!--img width="806" height="326" alt="image" src="https://github.com/user-attachments/assets/15c67450-0ed7-4deb-9375-3e7e04cb5f9b"-->
# Deep Dive into Gen AI & Large Language Models (Part 2)

**Lifecycle of LLM** <br>

<img width="900" height="512" alt="GenAI Part2" src="https://github.com/user-attachments/assets/3c5d2b41-6a6c-4764-9fa3-24e2328c1995" />

But before that we will Dive into Foundational Concepts.

## A **"Model** in Machine Learning
- A mathematical representation that has learned patterns from on new or unseen data without human intervention.
- This representation can be used to make either predictions, classifications, or decisions, based on the algorithm that was selected for the problem at hand.
- A model can process only numbers , so all the inputs - Text, image etc are transformed into numbers before processing.
- In case of Text , its converted into tokens using Natural Language Processing Algorithms

## NLP Quick Introduction: 
<img width="512" height="256" alt="image" src="https://github.com/user-attachments/assets/0b3e3441-4c7a-4ed8-a68b-ea9e74fc740e" />

## Tokenization Quick Introduction
- Is the process of breaking text into smaller units (tokens) that a model can understand and process.
- Tokens can be Characters, Subwords, Words or  Special symbols like punctuation, spaces.
- **Common Tokenization Methods**:
  - Whitespace / Word-based → split on spaces. Simple but fails with new/rare words.
  - Character-based → every character is a token. Very flexible but sequences get long.
  - Subword-based (most common today):
    - BPE (Byte Pair Encoding) → merges frequent character pairs.
    - WordPiece (used in BERT).
    - SentencePiece (used in GPT, T5, etc.).
- **Example: Sentence:"Unhappiness"**
  - Word-based → "Unhappiness" (one token, if in vocab).
  - Character-based → "U" "n" "h" "a" ... "s".
  - Subword (BPE) → "Un" "happiness".
  - Subword (WordPiece) → "Un" "##happiness".
- **In LLMs** Tokenization is done before input and after output like:
  - Input text → tokens → token IDs → embeddings → model.
  - Output IDs → tokens → text reconstruction.

## Embedding Quick Introduction & Visualization
- Visualization : [https://projector.tensorflow.org/](https://projector.tensorflow.org/)
- <img width="400" height="224" alt="image" src="https://github.com/user-attachments/assets/ce3b8bd9-6ec8-4cd1-b304-a25c2340696e" />

- Embeddings are **dense vector representations of tokens** in a continuous, multi-dimensional space.
- Each token (from the vocabulary) is mapped to a unique vector of numbers.
- please check the url 

Similar tokens have vectors that are closer together in this space.
## Tokenization Vs Embedding

<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/2af4b17a-066c-411f-8bc7-10c47b309ac8" />


## Overview of Lifecycle of LLM

### PRE-TRAINING PHASE
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/f7352630-2345-4830-8b74-4433c29ad43f" />

#### Vocabulary Creation
**Data Sources**
- A large corpus of text from public domain, licensed datasets (e.g., books, newswire) and synthetic data is collected.
- The data is preprocessed for cleaning, deduplication, filtering offensive/irrelevant text, remove PII etc.,
**Sampling**
- A representative sample is obtained full dataset (hundreds of billions of tokens).
- Sometimes sampling techniques are used to balance sources (so one source doesn’t dominate), to remove bias ([Refer](https://arxiv.org/abs/2407.11203))
- For very large corpora, subsampling may happen for efficiency, but the goal is broad coverage.
- 
**Tokenization**
- This sample is tokenized based on some algorithm (BPE, SentencePice or WordPiece etc.,)
- The idea is to split text into subword units that balance vocabulary size and flexibility (e.g., "unhappiness" → "un", "happiness").
- 
**Vocabulary**
- The output is a Vocabulary which has an Id for each Token which is tokenized
- Example: "cat" → ID 5021 , "##ting" → ID 7842 (continuation of a word), " " (space) → ID 320

### Embedding Matrix
- Now using the tokens from the Vocabulary an Embedding Matrix is created
  - Each token ID (integer) is mapped to a dense vector (embedding).
  - Initially these vectors are randomly initialized, not meaningful yet.
  - During training, embeddings learn semantic structure, (e.g., “king” – “man” + “woman” ≈ “queen”).
- This Embedding Matrix Size = [Vocabulary size × Embedding_dim]
  - Example: If vocab size = 50,000 and embedding_dim = 768 → matrix shape = (50,000 × 768),Each row corresponds to a token ID’s vector representation.
  - Embedding dimension depends on model size (and hardware): **GPT-2 small → 768, GPT-2 medium → 1024, GPT-3 → 12288**.
  - **Rule of thumb**: larger models → higher embedding dimensions → richer representations.

### Transformer Architecture & Training Objective

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/0360fc05-abeb-446a-969d-5c2b2f96e4ff" />


**Input Embeddings → Transformer Stack**
- Each token embedding (From Embedding Matrix) enters the Transformer encoder-decoder architecture (for LLMs like GPT, usually decoder-only).
- A positional encoding (sinusoidal or learned) is added so the model knows word order (since embeddings alone are orderless).
**Multi-Head Self-Attention**
- Attention answers: “Which other tokens in the sequence should this token pay attention to?”
- For each token, three vectors are computed: **Query (Q), Key (K), Value (V).**
- Attention score = similarity(Q, K) → determines how much weight each token assigns to others.
- **Multi-head** = multiple attention mechanisms in parallel, capturing different relationships (syntax, semantics, long vs short dependencies).

**Feed-Forward + Residual Connections**
- After **attention**, the outputs go through **Feed-forward Neural Networks** (two linear layers + nonlinearity).
- Residual connections + layer normalization stabilize the training.
- This stack (attention + feed-forward) is repeated **L** number of times (e.g., GPT-2 small → 12 layers, GPT-3 → 96 layers).

**Output Layer (Softmax over Vocabulary)**
- Final hidden states pass through a linear projection layer back to vocabulary size.
- Softmax gives a probability distribution over possible next tokens.

**Training Objective (Autoregressive Pre-training)**
- The model is trained to predict the next token given the previous ones.
- Example: Input: “The cat sat on the” , Target: “mat”
- Loss function = cross-entropy loss between predicted distribution and true next token.

**Optimization** 

Optimizer: usually AdamW with learning rate warmup + cosine decay.

Backpropagation adjusts weights (embeddings + attention + feed-forward layers) to reduce prediction error.

Training is distributed across thousands of GPUs/TPUs with parallelization techniques.

👉 So, Part 3 (Transformer + Training Objective) in short:

Token embeddings + positional encodings enter Transformer layers.

Multi-head self-attention learns token dependencies.

Feed-forward layers + normalization refine representations.

Output layer predicts next token probabilities.

Cross-entropy loss trains the model via backpropagation.



#### Vocabulary Creation 
#### Embedding Vector creation
### Training
### Fine-Tuning
### Deployment




## Natural Language Processing
## Embedding Process

## Embedding Algorithm





<a href="www.linkedin.com/in/tarriq-ferrose-khan-ba527080" target="_blank"><img width="406" height="226" alt="image" src="https://github.com/user-attachments/assets/15c67450-0ed7-4deb-9375-3e7e04cb5f9b"></a>

# Deep Dive into Gen AI & Large Language Models (Part 2)
<img width="900" height="512" alt="GenAI Part2" src="https://github.com/user-attachments/assets/3c5d2b41-6a6c-4764-9fa3-24e2328c1995" />

## Table Of Contents
[Quick Introduction of Foundational Concepts](https://github.com/tarriqferrosekhan/AI_KB/edit/main/AI_Model_Landscape/Part%202.md#quick-introduction-of-foundational-concepts)
- [A "Model" in Machine Learning](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#a-model-in-machine-learning)
- [NLP](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#nlp)
- [Tokenization](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#tokenization)
- [Embedding](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#embedding-with-visualization)
- [Tokenization Vs Embedding](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#tokenization-vs-embedding)
- [Neural Networks](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#neural-networks)
- [Transformer Encoder-Decoder Architecture](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#transformer-encoder-decoder-architecture)
  
[Life Cycle of LLM](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#lifecycle-of-llm)
- [PRE-TRAINING PHASE](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#pre-training-phase)
- [Embedding Matrix](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#embedding-matrix)
- [Transformer Architecture & Training Objective](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#transformer-architecture--training-objective)
- [Fine-Tuning & Alignment](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#fine-tuning--alignment)
- [Deployment & Inference](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Part%202.md#deployment--inference)


## Quick Introduction of Foundational Concepts:
### A **"Model"** in Machine Learning
- A mathematical representation that has learned patterns from on new or unseen data without human intervention.
- This representation can be used to make either predictions, classifications, or decisions, based on the algorithm that was selected for the problem at hand.
- A model can process only numbers , so all the inputs - Text, image etc are transformed into numbers before processing.
- In case of Text , its converted into tokens using **Natural Language Processing** Algorithms

### NLP: 
<img width="512" height="256" alt="image" src="https://github.com/user-attachments/assets/0b3e3441-4c7a-4ed8-a68b-ea9e74fc740e" />

### Tokenization
- Is the process of breaking text into smaller units (tokens) that a model can understand and process.
- Tokens can be Characters, Subwords, Words or  Special symbols like punctuation, spaces.
- **Common Tokenization Methods**:
  - Whitespace / Word-based → split on spaces. Simple but fails with new/rare words.
  - Character-based → every character is a token. Very flexible but sequences get long.
  - Subword-based (most common today):
    - **BPE** (Byte Pair Encoding) → merges frequent character pairs.
    - **WordPiece** (used in BERT).
    - **SentencePiece** (used in GPT, T5, etc.).
- **Example: Sentence:"Unhappiness"**
  - Word-based → "Unhappiness" (one token, if in vocab).
  - Character-based → "U" "n" "h" "a" ... "s".
  - Subword (BPE) → "Un" "happiness".
  - Subword (WordPiece) → "Un" "##happiness".
- **In LLMs** Tokenization is done before input and after output like:
  - Input text → tokens → token IDs → embeddings → model.
  - Output IDs → tokens → text reconstruction.

### Embedding (with Visualization)
- Visualization of Embeddings: [https://projector.tensorflow.org/](https://projector.tensorflow.org/)
- <img width="400" height="224" alt="image" src="https://github.com/user-attachments/assets/ce3b8bd9-6ec8-4cd1-b304-a25c2340696e" />
- Embeddings are **dense vector representations of tokens** in a continuous, multi-dimensional space.
- Each token (from the vocabulary) is mapped to a unique vector of numbers.
- **Embeddings capture semantic and syntactic relationships:"king" – "man" + "woman" ≈ "queen"**
- **Embedding Matrix** Size = Vocabulary size × Embedding dimension
- **In LLMs**:
  - Embeddings are the input layer: token IDs pointing to embedding vectors.
  - They serve as the foundation for all subsequent transformations (attention, feed-forward).
  - Output embeddings are also used before predicting the next token.

### Tokenization Vs Embedding

<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/2af4b17a-066c-411f-8bc7-10c47b309ac8" />

| Aspect      | Tokenization                                | Embedding                             |
| ----------- | ------------------------------------------- | ------------------------------------- |
| **Stage**   | First step (preprocessing)                  | After tokenization (representation)   |
| **Purpose** | Break text into tokens/IDs                  | Map tokens to meaningful vectors      |
| **Output**  | Tokens or IDs (discrete)                    | Dense vectors (continuous)            |
| **Analogy** | Like chopping a sentence into puzzle pieces | Like coloring each piece with meaning |

### Neural Networks 
- A neural network is a computer system that can learn complex patterns , inspired by how the human brain works.
- It learns patterns from data by passing information through layers of neurons (nodes).
- It has Layers of neurons (mathematical functions), Connected by weights (parameters) and Trained using backpropagation to minimize a **loss function**.
- Example:
  - **Input**: picture of a handwritten “5” -> **Hidden layers**: detect strokes, curves, shapes -> **Output**: predicts it’s the number “5”.
- **In LLMS**
  -LLMs use a type Neural Network called **Transformer Encoder-Decoder Architecture**

### Transformer Encoder-Decoder Architecture
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/e9be8aaa-995f-4863-af11-fa3487f79495" />

- The Transformer is a **Neural Network Design** that became the backbone of modern NLP.
- This was introduced in “[Attention is All You Need](https://arxiv.org/abs/1706.03762)”, 2017.
- It has two main parts:
  - **Encoder** → builds a representation that captures meaning + word relationships.
    - Input sentence is first **tokenized to embeddings + positional encoding**.
    - Each encoder layer has:
    - **Multi-head self-attention** which finds relationships between words.
    - **Feed-forward network** that processes info further.
    - Produces a set of context-rich vectors representing the input.
  - **Decoder**, Takes the encoder’s output + previously generated words.
    - Each decoder layer has:
      - **Masked self-attention** which ensures it only looks at previous words (can’t peek into the future).
      - **Encoder–decoder attention** which focuses on relevant parts of the input sentence.
      - **Feed-forward network** - **Attention** Layer figures out “who talks to who and this layer processes what each token does with that information.
    - Finally, a **softmax** layer predicts the next token.
      - After the Transformer layers process everything, NEXT TOKEN needs to be decided.
      - This Layer is what makes the model’s predictions interpretable—as probabilities—so we can pick the next token in text generation.
      - This layer gives probabilities for every word in the vocabulary, for Example:
        - “mat” → 0.80
        - “dog” → 0.15
        - “hat” → 0.05 , The model then samples or selects the next word.

## Lifecycle of LLM

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
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/6c0e2084-5514-46fc-bc90-8b091fd51681" />

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
- Optimizer: usually AdamW with learning rate warmup + cosine decay.
- Backpropagation adjusts weights (embeddings + attention + feed-forward layers) to reduce prediction error.
- Training is distributed across thousands of GPUs/TPUs with parallelization techniques.

### Fine-Tuning & Alignment
<img width="363" height="521" alt="image" src="https://github.com/user-attachments/assets/cfd58748-59e6-4dfa-bfba-fa1551009aaf" />

-  **Supervised Fine-Tuning (SFT)**
  -  Start with the pre-trained model.
  - Collect a curated dataset of instruction–response pairs (e.g., human-written Q&A, dialogues, reasoning tasks).
  - Train the model further to follow instructions better.
  - This makes the model more helpful and task-oriented (instead of just predicting raw next tokens).

- **Reinforcement Learning from Human Feedback (RLHF)**
  - Step 1: Humans rank multiple model outputs (for the same prompt).
  - Step 2: Train a reward model to predict these rankings.
  - Step 3: Use reinforcement learning (PPO, Proximal Policy Optimization) to optimize the LLM so that it produces outputs closer to human preference.
  - **Purpose**: reduce harmful, irrelevant, or nonsensical answers.

- **Guardrails & Safety Layers**:
  - Add filters, policies, and moderation layers outside the model.
  - Examples:
    - Refuse unsafe instructions (e.g., violence, self-harm).
    - Mask sensitive PII (personal info).
    - Ensure factual grounding (RAG, citations).
    - These aren’t part of the model weights, but system-level enforcement.

- **Other Adaptation Techniques**
- Instruction tuning → train on datasets with explicit instructions.
- Domain-specific fine-tuning → adapt the model for medicine, law, finance, etc.
- Parameter-efficient fine-tuning (PEFT) → methods like LoRA, adapters, prefix-tuning so fine-tuning can be done cheaply without retraining billions of parameters.

 **After this stage, the model becomes not just a language predictor, but a helpful AI assistant that follows human values, instructions, and safety norms.**


### Deployment & Inference
<img width="521" height="363" alt="image" src="https://github.com/user-attachments/assets/e51def9e-f115-4b5f-bc21-a83aaac50194" />

- **Model Serving**:
  - The trained model weights (often hundreds of GBs) are deployed on specialized hardware (GPUs, TPUs, or custom AI accelerators).
  - Techniques used:
    - Model parallelism which splits model layers across multiple GPUs.
    - Tensor parallelism which split tensor computations across GPUs.
    - Quantization (e.g., FP32 → INT8/INT4) to reduce memory + speed up inference.
    - Sharding + caching for distributed inference.
- **Input Processing (Tokenization)**
  - User input are tokenized using the same tokenizer from Pre-training (BPE, SentencePiece, etc.).
  - Each token ID is mapped to its embedding vector from the learned embedding matrix.
- **Autoregressive Generation**:
  - The input embeddings go through the Transformer layers.
  - The model outputs a probability distribution over the vocabulary for the next token.
- **Sampling strategies**:
  - Greedy decoding → pick top-1 token each step (deterministic, but boring).
  - Beam search → explores multiple likely continuations.
  - Top-k / Top-p (nucleus) sampling → adds randomness for creativity.
  - This repeats token by token until end-of-sequence or user stop.
- **Post-Processing**:
  - Token IDs → converted back to text.
  - Extra layers (outside the model) may:
    - Filter unsafe outputs (guardrails).
    - Add citations (RAG systems).
    - Integrate with external tools/APIs

- **Integration into Applications**:
- LLMs can be used directly (ChatGPT-style assistants) or as building blocks inside bigger systems:Chatbots / Agents (customer support, productivity).
- RAG (Retrieval-Augmented Generation) → grounding answers in external databases.
- Coding Assistants → Copilot, code completion.
- Enterprise Applications → summarization, search, automation.
 
(To be Continued)

Happy Learning
Author : [Tarriq Ferrose Khan](www.linkedin.com/in/tarriq-ferrose-khan-ba527080) 


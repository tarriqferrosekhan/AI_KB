#Author : [Tarriq Ferrose Khan](www.linkedin.com/in/tarriq-ferrose-khan-ba527080) 

<!--img width="806" height="326" alt="image" src="https://github.com/user-attachments/assets/15c67450-0ed7-4deb-9375-3e7e04cb5f9b" /-->


# Deep Dive into Gen AI & Large Language Models (Part 2)

**In this part we will see **Lifecycle of LLM** 
But before that we will Dive into Foundational Concepts.

## A **"Model** in Machine Learning
- A mathematical representation that has learned patterns from on new or unseen data without human intervention.
- This representation can be used to make either predictions, classifications, or decisions, based on the algorithm that was selected for the problem at hand.

## NLP Basics: 
### Lexical
### Syntacital
### Semantical 
### Pragmatic

## Tokenization Vs Embedding


## Overview of Lifecycle of LLM

### PRE-TRAINING PHASE
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
**Vocabulary**
- The output is a Vocabulary which has an Id for each Token which is tokenized
- Example: "cat" → ID 5021 , "##ting" → ID 7842 (continuation of a word), " " (space) → ID 320

#### Embedding
****


#### Vocabulary Creation 
#### Embedding Vector creation
### Training
### Fine-Tuning
### Deployment




## Natural Language Processing
## Embedding Process

## Embedding Algorithm





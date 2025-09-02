#Author : [Tarriq Ferrose Khan](www.linkedin.com/in/tarriq-ferrose-khan-ba527080) 

<img width="806" height="326" alt="image" src="https://github.com/user-attachments/assets/15c67450-0ed7-4deb-9375-3e7e04cb5f9b" />


# Deep Dive into Large Language Models

**Topics**<br>
[LLM Classification Definition](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Readme.md#llm-classification-definition)<br>
[Models Types -Use Case n Hosting](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Readme.md#ai-models-types--use-case-n-hosting)<br>
[AI Tools List By Xploria](https://github.com/tarriqferrosekhan/AI_KB/blob/main/AI_Model_Landscape/Readme.md#ai-tools-by-xploria-httpswwwaixploriacomenultimate-list-ai)<br>

## Gen AI Introduction
- AI is focused on creating systems that mimic human intelligence.
- Machine Learning (ML) is a subset of AI which learns patterns from underlying data using specific algorithms.
- ML algorigthm can be classified majorly into:
    -    **Discriminative**: Helps to Learn from Underlying data to classify or Predict.
    -    **Generative** : Helps to generate/replicate basis the underlying distribution of data.
- **Generative Algorigthms** can be classified into Classical & **Modern**
## Modern Algorithms
- [Variational Auto Encoders](https://www.ibm.com/think/topics/variational-autoencoder)
- [Generative Adverserial Networks](https://aws.amazon.com/what-is/gan/)
- [Diffusion Models](https://www.ibm.com/think/topics/diffusion-models)
- [Autoregressive Models](https://aws.amazon.com/what-is/autoregressive-models/) - generate outputs one token at a time, each step depending on the previously generated tokens.
- **Most of the Gen AI models (not all) are based on AutoRegressive Algorithms**

### Large Language Model (LLM) Introduction
- Is an AI system **Pre-trained** on vast amounts of text data that learns the statistical relationships between words and phrases.
- Fine-tuned to follow instructions, be safe, and helpful.
- It predicts the most likely next word (or sequence of words), enabling it to generate human-like text, answer questions, translate, summarize, and even reason.
- Examples: GPT (OpenAI), Claude (Anthropic), Gemini (Google DeepMind), LLaMA (Meta).

<img width="256" height="256" alt="LLM Overview" src="https://github.com/user-attachments/assets/e69e29f0-65a2-43bb-bf68-b558940d9713" />

### GPTs Vs LLM
- Generative Pre-Trained Transformers  are specific instance of LLMs
- In short : All GPTs are LLMs, but not all LLMs are GPTs.

| **Aspect**       | **LLM (Large Language Model)**                                                                       | **GPT (Generative Pretrained Transformer)**                                           |
| ---------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Definition**   | Any AI model trained on massive amounts of text data to understand and generate human-like language. | A specific family of LLMs developed by OpenAI, based on the Transformer architecture. |
| **Scope**        | General term – includes GPT, Claude, LLaMA, PaLM, Gemini, Mistral, etc.                              | A subset of LLMs – GPT-1, GPT-2, GPT-3, GPT-3.5, GPT-4, GPT-5.                        |
| **Architecture** | Most are Transformer-based, but LLM is a broader concept (not tied to one vendor).                   | Strictly Transformer-based with autoregressive training.                              |
| **Training**     | Trained on huge text corpora, sometimes multimodal (text, images, code, audio).                      | Pretrained on diverse internet-scale datasets, optimized by OpenAI.                   |
| **Use Cases**    | Chatbots, translation, summarization, coding assistants, knowledge retrieval, reasoning tasks.       | ChatGPT, Copilot, Codex, and other OpenAI products/services.                          |
| **Ownership**    | Open source (LLaMA, Falcon, Mistral, etc.) + proprietary (Claude, Gemini, GPT).                      | Proprietary by OpenAI (though older GPT models are partially open).                   |
| **Analogy**      | LLM is like saying “car” (general category).                                                         | GPT is like saying “Tesla” (a specific brand of car within the category).             |

### Transformer Model Overview
- A Transformer is a type of neural network architecture introduced in the paper “[Attention is All You Need](https://arxiv.org/abs/1706.03762)” (2017).
- Old Models are sequence models - RNNs, LSTMs, or CNNs and were slow in training, hard to parallelize.
- Transformer Model removes recurrence entirely, relies only on **self-attention mechanisms** to process sequences, making training faster, parallelizable, and effective at capturing long dependencies.
<img width="256" height="512" alt="Transformer Models" src="https://github.com/user-attachments/assets/fd92ea07-a5de-4365-9380-90a6f14026ad" />


### Tokens in an LLM
- A token is a small unit of text that an LLM processes.
- The model doesn’t see words, it sees tokens.
- Each token is mapped into a high-dimensional vector space called **Embeddings**
- **Context Length**: LLMs can only “remember” a limited number of tokens in one go

### Parameters in an LLM
- Parameters are the weights & biases of the neural network.
- They define how input tokens are transformed into meaningful output. 
- **Parameter Generation Process steps**:
    - **Initialization**: Initially They are set as Random numbers , Example: In GPT-3 (175B parameters), all 175 billion weights start as random values.
    - **Training**:
        - Input tokens → embeddings → pass through Transformer layers.
        - Model predicts the next token (autoregressive), compares prediction with the actual token (from dataset) and Compute loss (error).
        - Use backpropagation + [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) to adjust weights.
        - After a trillions of steps, the parameters shift from random values to values that actually capture grammar, facts, reasoning patterns, etc.
- **Fine-Tuning** (Optional)
    - Base LLM (foundation model) can be further tuned:
    - Instruction tuning (makes it follow human commands).
    - RLHF (Reinforcement Learning with Human Feedback) (aligns responses to preferences).
    - Domain-specific tuning (e.g., legal, medical).

## Types of Parameters in an LLM
| **Category**                                        | **Examples**                                                        |
| --------------------------------------------------- | ------------------------------------------------------------------- |
| **Model Parameters** (learned)                      | Weights, biases, embeddings, attention matrices                     |
| **Hyperparameters** (fixed at training)             | Layers, hidden size, attention heads, context length, learning rate |
| **Decoding Parameters** (user-defined at inference) | Temperature, Top-K, Top-P, max tokens, repetition penalty           |

### Tokens Vs Parameters
- **Summary** :
- Tokens are the words you fed, **Parameters** are the “brain wiring” that decides what to do with those words.
- The more parameters, the richer the understanding of how tokens relate.
- **Parameters are stored “memory” of how tokens typically appear together.**
- **Process Steps**:
    - Each token is converted to an [embedding vector](https://www.pinecone.io/learn/vector-embeddings/).
    - The model’s parameters (weights in attention & feedforward layers) transform these embeddings.
    - Parameters decide how the current token relates to previous tokens.
    - During **training**, the model sees billions of tokens and adjusts its parameters to minimize prediction error (next-token prediction).

### MultiModal (Image,Audio,Video) Processing by LLMs
- **Images**
    - As an image is not made of words, it is tokenized into small patches (e.g., 16×16 pixels) or discrete visual tokens.
    - Once the image patches/tokens are created, they are processed by a transformer encoder (or encoder–decoder).
    - Self-attention lets the model learn relationships between different parts of the image (e.g., “eye” ↔ “face,” “wheel” ↔ “car”).
    - In case of  Multimodal LLMs, (like GPT-4V, LLaVA, Kosmos-1, Gemini) - The image embeddings are combined with text embeddings.
    - [Cross-attention layers](https://arxiv.org/html/2502.02406v1) allow the model to align visual tokens with text tokens.
    - Example: Question: “What is in the picture?” → The image tokens guide the text decoder to produce “A dog sitting on a sofa.”
- Similarly **Audio** is transformed into Spectrograms - Convert audio into a 2D time–frequency map and **Video** into sequence of images (frames) + optional audio track

### Model Capacity Vs Model Size

| **Aspect**        | **Model Size**                                                                            | **Model Capacity**                                                                                                                    |
| ----------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Definition**    | The total number of parameters in the model (e.g., 7B, 70B, 175B). It’s a *raw count*.    | The model’s actual ability to learn, generalize, and represent complex patterns from data.                                            |
| **Analogy**       | 💾 **Storage space** on a computer. More gigabytes doesn’t always mean faster or smarter. | 🧠 **Cognitive ability** — how effectively the system uses its storage + processing.                                                  |
| **Influenced By** | - Number of parameters<br>- Embedding size<br>- Layers                                    | - Model size<br>- Architecture efficiency (e.g., Transformer vs RNN)<br>- Training data quality & diversity<br>- Optimization methods |
| **Measurement**   | Easy to quantify (just count parameters).                                                 | Harder to measure directly — evaluated via benchmarks, reasoning ability, generalization.                                             |
| **Limitations**   | Bigger ≠ always better (may overfit, inefficient, costly).                                | A smaller but well-trained model (good data + architecture) can have **higher effective capacity** than a larger poorly trained one.  |
| **Example**       | GPT-3 = 175B parameters → large size.                                                     | LLaMA-2 13B often outperforms GPT-3 175B on reasoning → higher *effective capacity*.                                                  |

### Model Cards
- A Model Card is a structured documentation template (proposed by Google AI in 2019) that provides key details about an ML model’s development, intended use, performance, risks, and limitations.
- **It’s like a “nutrition label” for AI models.**

### Responsible AI vs Explainable AI

| Aspect           | Responsible AI                          | Explainable AI                         |
| ---------------- | --------------------------------------- | -------------------------------------- |
| **Definition**   | Ethical + safe AI development and use   | Making AI decisions understandable     |
| **Focus**        | Governance, fairness, accountability    | Interpretability, transparency         |
| **Scope**        | Broad (policy, ethics, law, operations) | Narrower (technical model insights)    |
| **Who benefits** | Society, policymakers, organizations    | End-users, developers, regulators      |
| **Example**      | AI avoids racial bias in hiring         | AI explains why candidate was selected |


## LLM Classifications
<img width="512" height="512" alt="ChatGPT Image Sep 2, 2025, 10_11_34 PM" src="https://github.com/user-attachments/assets/bfdeb219-8fed-400d-96f4-873668a51452" />

### LLM Classification Definition:

| **Category**          | **Definition**                                                                                                                                                         | **Analogy**                                                                                      |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Foundation Models** | Large, general-purpose models trained on massive, diverse datasets. They act as the “base layer” of intelligence and can be adapted to many downstream tasks.          | 🧠 **Like a raw brain** — powerful, flexible, but needs direction or training for specific jobs. |
| **Industry Models**   | Proprietary, production-ready foundation models offered by companies. Optimized for scale, safety, and enterprise use, but typically only accessible via API or cloud. | 🏢 **Like a finished product** — polished, packaged, and ready for business use.                 |
| **Task Models**       | Specialized models fine-tuned or designed for specific domains or tasks (e.g., coding, vision, speech, science). Often derived from foundation models.                 | 👨‍🔧 **Like a skilled specialist** — trained to excel at a specific task.                       |


### Models Types -Use Case n Hosting:
<img width="512" height="512" alt="ChatGPT Image Sep 2, 2025, 09_47_21 PM" src="https://github.com/user-attachments/assets/a28fd459-acb1-44af-b863-642a93d357eb" />

<table border=0 style="background-color:red">
<tr>
    <td><small><b>Category</b></small></td>
    <td><b>Examples</b></td>
    <td> <b>Type</b></td>
    <td><b>Typical Use Cases</b></td>
    <td><b>Local Hosting?</b></td>
    <td><b>Recommended Frameworks</b></td>
</tr>
<tr>
  <td> 
      <b>Foundation Models (General Purpose, Open Source)</b> 
  </td>
  <td>
      <a href="https://www.llama.com/" target="_blank">LLaMA 2/3 (Meta)</a><br>
      <a href="https://mistral.ai/news/mixtral-of-experts"  target="_blank">Mistral/Mixtral</a><br>
      <a href="https://falconllm.tii.ae/falcon-180b.html" target="_blank">Falcon 40B/180B</a><br>
      <a href="https://bigscience.huggingface.co/blog/bloom" target="_blank">BLOOM</a>
      <a href="https://deepmind.google/models/gemma/" target="_blank">Gemma (Google)</a><br>
      <a href="https://www.databricks.com/blog/mpt-7b">MPT (Databricks)</a>         
      </td>
      <td> Open Source</td>
      <td> Text generation, RAG, fine-tuning</td>
      <td> ✅ Yes</td>
      <td> **Ollama**, **vLLM**, **llama.cpp**, Hugging Face **TGI**, **Text Generation WebUI**</td>
      </tr>
      <tr>
        <td> <b>Industry Foundation Models (Proprietary)</b></td>
        <td> 
          GPT-4o, GPT-3.5 (<a href="https://platform.openai.com/docs/models" target="_blank">OpenAI</a>)<br>
          <a href="https://claude.ai/" target="_blank">Claude 3<a> (<a href="https://www.anthropic.com/" target="_blank">Anthropic</a>), <a href="https://arxiv.org/abs/2403.05530" target="_blank">Gemini 1.5 (Google)</a><br>
            <a href="https://cohere.com/command" target="_blank">Cohere Command R+</a><br>
            <a href="https://www.ibm.com/granite/playground/" target="_blank">IBM Granite</a> 
        </td>
        <td> Proprietary </td>
        <td> Copilots, enterprise automation, chatbots</td>
        <td> ❌ No (API/cloud only)</td>
        <td> N/A (hosted via API, e.g. Azure, AWS Bedrock)</td>
      </tr>
      <tr>
            <td> <b>Task Models – Code</b></td>
            <td> <a href="https://ai.meta.com/blog/code-llama-large-language-model-coding/" target="_blank">Code LLaMA (Meta)</a><br>
                 <a href="https://mistral.ai/news/codestral" target="_blank">Codestral (Mixtral)</a><br>
                 <a href="https://huggingface.co/blog/starcoder" target="_blank">StarCoder (Hugging Face)</a><br>
                 <a href="https://ollama.com/library/wizardcoder" target="_blank">WizardCoder</a><br>
                 PolyCoder
            </td>
            <td> Mix</td>
            <td> Code generation, debugging</td>
            <td> ✅ Yes (for open-source ones)</td>
            <td> Ollama, vLLM, Hugging Face TGI</td>
      </tr>
      <tr><td> <b>Task Models – Vision</b></td>
        <td> <a href="https://stablediffusionweb.com/">Stable Diffusion</a>, Meta SAM, Google Imagen, DINOv2, DALL·E</td>
        <td> Mix         </td>
        <td> Image generation, segmentation, object detection </td>
        <td> ✅ Yes (Stable Diffusion, SAM, DINOv2); ❌ Imagen, DALL·E </td>
        <td> 
          <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">Automatic1111</a><br>
          <a href="https://www.comfy.org/">ComfyUI</a><br>
          <a href="https://huggingface.co/docs/diffusers/en/index">Hugging Face Diffusers</a><br>
        </td>
    </tr>
    <tr>
          <td> <b>Task Models – Speech/Audio</b></td>
          <td> Whisper (OpenAI), Meta MMS, WaveNet, AudioLM, ElevenLabs TTS</td>
          <td> Mix</td>
          <td> Speech-to-text, transcription, TTS</td>
          <td> ✅ Whisper, MMS; ❌ WaveNet, AudioLM, ElevenLabs</td>
          <td> Whisper.cpp, <a href="https://speechbrain.github.io/">SpeechBrain</a>, Hugging Face Transformers</td>
    </tr>
    <tr>
          <td> <b>Task Models – Multimodal</b></td>
          <td> Kosmos-2 (MSR), IDEFICS (Hugging Face), GPT-4o, Gemini 1.5</td>
          <td> Mix</td>
          <td> Image + text + audio assistants                  </td>
          <td> ✅ Kosmos-2, IDEFICS; ❌ GPT-4o, Gemini         </td>
          <td> Hugging Face TGI, **Ollama (multi-modal support emerging)**</td>
    </tr>
    <tr>
        <td> <a>Task Models – Science/Medicine</a></td>
        <td> AlphaFold (DeepMind), BioGPT, Med-PaLM 2</td>
        <td> Mix</td>
        <td> Protein folding, biomedical NLP</td>
        <td> ✅ BioGPT; ⚠️ AlphaFold (very GPU-heavy); ❌ Med-PaLM 2</td>
        <td> AlphaFold GitHub release, Hugging Face Transformers</td>
    </tr>
    <tr>
        <td> <b>Task Models – Embeddings / Retrieval</b></td>
        <td> Sentence-BERT, E5, OpenAI Embeddings, Cohere Embeddings</td>
        <td> Mix</td>
        <td> Semantic search, RAG, clustering</td>
        <td> ✅ SBERT, E5; ❌ OpenAI/Cohere embeddings</td>
        <td> SentenceTransformers (Python), Hugging Face Inference, LangChain + LocalEmbeddings ,LLAMAIndex </td>
    </tr>
</table>

### AI Tools by XPloria: https://www.aixploria.com/en/ultimate-list-ai/

**To be Continued**

Happy Learning 
[Tarriq Ferrose Khan](www.linkedin.com/in/tarriq-ferrose-khan-ba527080)


# Deep Dive into Gen AI

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

### Tokens in an LLM
- A token is a small unit of text that an LLM processes.
- The model doesn’t see words, it sees tokens.
- Each token is mapped into a high-dimensional vector space called **Embeddings**
- **Context Length**: LLMs can only “remember” a limited number of tokens in one go

### Parameters in an LLM
- Parameters are the weights & biases of the neural network.
- They define how input tokens are transformed into meaningful output. 
- Process steps:
    - Initially They are set as Random numbers , Example: In GPT-3 (175B parameters), all 175 billion weights start as random values.
    - Training:
        - Input tokens → embeddings → pass through Transformer layers.
        - Model predicts the next token (autoregressive), compares prediction with the actual token (from dataset) and Compute loss (error).
        - Use backpropagation + [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) to adjust weights.
        - After a trillions of steps, the parameters shift from random values to values that actually capture grammar, facts, reasoning patterns, etc.
- Fine-Tuning (Optional)
    - Base LLM (foundation model) can be further tuned:
    - Instruction tuning (makes it follow human commands).
    - RLHF (Reinforcement Learning with Human Feedback) (aligns responses to preferences).
    - Domain-specific tuning (e.g., legal, medical).

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



## AI Model Classification
<img width="512" height="512" alt="ChatGPT Image Sep 2, 2025, 10_11_34 PM" src="https://github.com/user-attachments/assets/bfdeb219-8fed-400d-96f4-873668a51452" />

### Classification Definition:

| **Category**          | **Definition**                                                                                                                                                         | **Analogy**                                                                                      |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Foundation Models** | Large, general-purpose models trained on massive, diverse datasets. They act as the “base layer” of intelligence and can be adapted to many downstream tasks.          | 🧠 **Like a raw brain** — powerful, flexible, but needs direction or training for specific jobs. |
| **Industry Models**   | Proprietary, production-ready foundation models offered by companies. Optimized for scale, safety, and enterprise use, but typically only accessible via API or cloud. | 🏢 **Like a finished product** — polished, packaged, and ready for business use.                 |
| **Task Models**       | Specialized models fine-tuned or designed for specific domains or tasks (e.g., coding, vision, speech, science). Often derived from foundation models.                 | 👨‍🔧 **Like a skilled specialist** — trained to excel at a specific task.                       |


### AI Models Types -Use Case n Hosting:
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

### Comprehensive set of AI Tools by XPloria: https://www.aixploria.com/en/ultimate-list-ai/

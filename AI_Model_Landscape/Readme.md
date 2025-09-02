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

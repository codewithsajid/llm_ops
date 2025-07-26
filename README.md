# LLM-Ops: Agentic RAG + LLM Chatbot

A cutting-edge framework implementing **Retrieval-Augmented Generation (RAG)** combined with an **agentic decision-making** component powered by **Large Language Models (LLMs)**. Designed for scalability, quality, and ease of operations (LLMOps), this system integrates local knowledge bases and real-time web search to provide accurate, relevant, and dynamic conversational responses.

---

## 🌟 Key Features

### ✅ **Agentic Retrieval-Augmented Generation (RAG)**

* Dynamically evaluates the sufficiency of local knowledge.
* Automatically decides when to augment responses with fresh web data.
* Ensures responses are timely, accurate, and contextually relevant.

### ✅ **Advanced LLM Integration**

* Utilizes Google's **Gemma 7B** LLM for high-quality conversational outputs.
* Supports customizable generation settings (temperature, sampling).

### ✅ **Smart Retrieval & Re-ranking**

* **Hybrid search** via Weaviate for vector-based semantic retrieval.
* **CrossEncoder re-ranking** ensures precision and relevance.
* Efficient deduplication and token budgeting prevent prompt overflow.

### ✅ **Web Search Integration**

* Leverages DuckDuckGo API for real-time web searches.
* LLM-generated smart queries ensure accurate and relevant search results.
* Content extraction powered by Trafilatura ensures clean text inputs.

### ✅ **Robust Evaluation & Decision Making**

* Intelligent confidence-based evaluation by LLM.
* Structured reasoning prompts for reliable decision-making.

### ✅ **LLMOps & Observability**

* MLflow integration for logging and experiment tracking.
* Rich, structured logging for debugging and operational insights.

---

## Architecture Overview

```
User Question
      │
      ▼
┌───────────────┐      ┌───────────┐      ┌──────────────┐
│ Retrieval     │─────▶│ Evaluator │─────▶│ Local KB     │
└───────────────┘      └───────────┘      └──────────────┘
      │                        │
      ▼                        ▼
┌───────────────┐      ┌───────────────┐
│ Web Searcher  │◀────▶│ Decision Gate │
└───────────────┘      └───────────────┘
      │
      ▼
┌───────────────┐      ┌───────────┐
│ Reranker      │─────▶│ LLM Chat  │
└───────────────┘      └───────────┘
      │
      ▼
Structured Response
(with citations)
```

---

## 🚩 Quickstart

### Installation

```bash
git clone https://github.com/codewithsajid/llm_ops.git
cd llm_ops
pip install -r requirements.txt
```

### Usage

Run the chatbot locally:

```bash
python -m llm_ops.rag_chatbot --question "What are the latest developments in RL?" --web --creative
```

### Additional Flags

* `--web`: Enables agentic web search.
* `--creative`: Uses creative response generation (temperature = 0.7).
* `--debug`: Enables detailed debug outputs.

---

## 📖 Project Structure

```
llm_ops/
├── agents/
│   └── rag_agent.py
├── llm/
│   └── gemma.py
├── utils/
│   ├── web_search.py
│   └── query_generator.py
├── weaviate_client.py
├── rag_chatbot.py
├── mlflow_utils.py
└── prompt_templates.py
```

---

## 🔮 Future Roadmap

* **Multimodal RAG**: Integrate image and video retrieval.
* **Enhanced Agentic Behavior**: Incorporate advanced planning and reasoning loops.
* **Interactive UI**: Web-based chatbot interface with visualizations.

---

## 📃 Citation

Please cite this repository if used in your research or projects:

```
@misc{llmops2025,
  author = {Sajid Ansari},
  title = {LLM-Ops: Agentic Retrieval-Augmented Generation with LLM Chatbot},
  year = {2025},
  url = {https://github.com/codewithsajid/llm_ops}
}
```

---

## 🌐 Contributing

Contributions and feedback are highly encouraged! Open an issue or submit a pull request to improve the project.

---

## 📬 Contact

* GitHub: [codewithsajid](https://github.com/codewithsajid)
* Email: [your.email@example.com](mailto:your.email@example.com)

---

Happy RAGging! 🚀✨

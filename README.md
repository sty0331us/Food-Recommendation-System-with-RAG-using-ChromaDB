# Food Recommendation System with RAG using ChromaDB

Semantic food recommendations powered by ChromaDB vector search and IBM watsonx.ai (Granite), with retrieval-only fallback when the LLM is unavailable.

## Features

- Persistent ChromaDB collections with sentence-transformer embeddings
- Hybrid search CLI: similarity plus cuisine / calorie filters
- Enhanced RAG chatbot with contextual explanations and query comparison
- Environment-driven configuration via `.env`

## Requirements

- Python 3.10+ (3.11 preferred)
- Optional: IBM watsonx.ai credentials for Granite generation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` if you need a personal watsonx API key. For IBM Skills Network labs, `WATSONX_PROJECT_ID=skills-network` often works without a personal key.

## Usage

### Enhanced RAG chatbot

```bash
python enhanced_rag_chatbot.py
# or
python food_recommendation_system_with_rag.py
```

Useful flags: `--rebuild-index`, `--top-k N`, `--skip-llm-check`.

Interactive commands: `help`, `compare`, `quit`.

### Advanced hybrid search

```bash
python advanced_search.py
python advanced_search.py --query "spicy Italian pasta" --cuisine Italian --max-calories 400
```

## Project layout

```
food_rag/           # config, ChromaDB, LLM, RAG pipeline
data/               # FoodDataSet.json
advanced_search.py  # hybrid search CLI
enhanced_rag_chatbot.py
shared_functions.py # backward-compatible imports
```

## License

MIT — see [LICENSE](LICENSE).

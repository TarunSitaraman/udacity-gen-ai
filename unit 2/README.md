# NASA RAG Chat — Unit 2 Project

A Retrieval-Augmented Generation (RAG) system for querying NASA space mission documents (Apollo 11, Apollo 13, Challenger) with real-time RAGAS evaluation.

## Project Structure

```
unit 2/
├── embedding_pipeline.py   # Chunks text files and stores embeddings in ChromaDB
├── rag_client.py           # ChromaDB retrieval and context formatting
├── llm_client.py           # OpenAI chat completion wrapper
├── ragas_evaluator.py      # RAGAS response-quality evaluation
├── chat.py                 # Streamlit chat application
├── batch_eval.py           # Batch evaluation runner (prints per-question + mean metrics)
├── test_questions.json     # Evaluation dataset (7 questions, all three missions)
├── requirements.txt        # Python dependencies
└── data_text/
    ├── apollo11/           # Apollo 11 transcript and document text files
    ├── apollo13/           # Apollo 13 transcript and document text files
    └── challenger/         # Challenger STS-51L audio transcript text files
```

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set API key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Pipeline

### Step 1 — Build the vector database

Process all text files in `data_text/` and store embeddings in ChromaDB:

```bash
python embedding_pipeline.py \
  --openai-key "$OPENAI_API_KEY" \
  --data-path ./data_text \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text
```

Check collection statistics without reprocessing:

```bash
python embedding_pipeline.py \
  --openai-key "$OPENAI_API_KEY" \
  --chroma-dir ./chroma_db_openai \
  --stats-only
```

### Step 2 — Launch the chat interface

```bash
streamlit run chat.py
```

The sidebar lets you select the ChromaDB collection, enter your API key, choose a model, and enable/disable RAGAS evaluation per response.

### Step 3 — Run batch evaluation

Evaluate all questions in `test_questions.json` and print per-question RAGAS metrics plus the mean per metric:

```bash
python batch_eval.py \
  --openai-key "$OPENAI_API_KEY" \
  --chroma-dir ./chroma_db_openai \
  --collection nasa_space_missions_text
```

Optional flags:
- `--questions path/to/questions.json` — use a different evaluation dataset
- `--model gpt-4` — change the answer-generation model (default: gpt-3.5-turbo)
- `--n-docs 5` — number of documents retrieved per question (default: 3)

## Evaluation Dataset

`test_questions.json` contains 7 questions covering all three missions and multiple document categories:

| # | Mission | Category |
|---|---------|----------|
| 1 | Apollo 11 | mission_report |
| 2 | Apollo 13 | technical |
| 3 | Apollo 11 | public_affairs_officer |
| 4 | Apollo 13 | technical |
| 5 | Apollo 11 | command_module |
| 6 | Challenger | mission_audio |
| 7 | Apollo 11 | flight_plan |

## RAGAS Metrics

The evaluator reports:

- **response_relevancy** — how well the answer addresses the question
- **faithfulness** — whether the answer is grounded in the retrieved context

## Component Quick-Test

```python
# Test RAG retrieval
from rag_client import discover_chroma_backends
print(discover_chroma_backends())

# Test evaluation input validation
from ragas_evaluator import evaluate_response_quality
print(evaluate_response_quality("", "answer", ["ctx"]))
# → {"error": "question must be a non-empty string"}
```

## Metadata Keys

Documents stored in ChromaDB use these metadata fields:

| Field | Description |
|-------|-------------|
| `mission` | `apollo_11`, `apollo_13`, or `challenger` |
| `document_category` | `public_affairs_officer`, `command_module`, `technical`, `flight_plan`, `mission_audio`, `nasa_archive`, etc. |
| `source` | Filename stem |
| `data_type` | `transcript`, `audio_transcript`, `flight_plan`, `document` |
| `chunk_index` | Position of chunk within the source file |

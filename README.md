# AutoResearch AI

> AutoResearch AI is an automated AI research agent designed to continuously research user-defined topics, build a persistent knowledge base, generate source-grounded reports, and identify new developments over time.

---

## Current Status

```
Stage 1 — Project Foundation          ✅
Stage 2 — Google Gemini Integration   ✅
Stage 3 — Automated Source Discovery  ✅
Stage 4 — Document Processing + RAG   ✅
Stage 5 — Research Agent + Reports    ✅
Stage 6 — Persistent Memory + What's New ✅
Stage 7 — Scheduling + Automation     ✅
Stage 8 — FastAPI Backend + Streamlit UI ✅
```

---

## Planned Capabilities

> **Note:** The features listed below are planned for future stages and are **not yet implemented**.

| Feature | Description |
|---|---|
| 🔍 Automated source discovery | Searches the web for relevant sources on a given topic |
| 📄 Document processing | Extracts and cleans text from web pages and documents |
| 🧠 Embeddings | Converts documents into semantic vector representations |
| 🗄️ Vector database | Stores and indexes embeddings for efficient retrieval |
| 📚 Retrieval-Augmented Generation (RAG) | Retrieves relevant context before generating answers |
| 🤖 LLM research synthesis | Uses a large language model to generate structured reports |
| 📎 Source citations | Every claim in a report is linked to a source |
| 🕑 Research history | Stores all past research runs for comparison |
| 🆕 What's New detection | Identifies developments not present in previous research |
| ⏰ Scheduled research automation | Runs research automatically on a configurable schedule |
| 📊 Research dashboard | A web UI to view reports, history, and insights |

---

## Project Structure

```text
autoresearch-ai/
│
├── app/
│   ├── __init__.py        # Python package marker
│   ├── main.py            # Application entry point & startup logic
│   ├── config.py          # Centralized configuration (loaded from .env)
│   │
│   └── llm/
│       ├── __init__.py    # LLM sub-package marker
│       └── client.py      # Reusable Google Gemini API client
│
├── data/
│   └── .gitkeep           # Placeholder — raw data stored here (gitignored)
│
├── reports/
│   └── .gitkeep           # Placeholder — generated reports stored here (gitignored)
│
├── tests/
│   ├── __init__.py        # Python package marker
│   ├── test_config.py     # Stage 1 configuration tests
│   └── test_llm.py        # Stage 2 Gemini client tests (mocked)
│
├── .env                   # Local environment variables (NOT committed)
├── .env.example           # Template for environment variables
├── .gitignore             # Standard Python gitignore
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── run.py                 # Root-level application launcher
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/autoresearch-ai.git
cd autoresearch-ai
```

### 2. Create and activate a virtual environment (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Configuration is managed through a `.env` file in the project root.

1. Copy the example file:

   ```bash
   copy .env.example .env
   ```

2. Edit `.env` with your values:

   | Variable | Required | Description |
   |---|---|---|
   | `APP_NAME` | No | Application display name |
   | `APP_VERSION` | No | Application version |
   | `ENVIRONMENT` | No | `development` \| `staging` \| `production` |
   | `LOG_LEVEL` | No | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
   | `GEMINI_API_KEY` | **Yes (Stage 2+)** | Your Google Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com/apikey) |
   | `GEMINI_MODEL` | No | Model name, default: `gemini-2.5-flash-lite` |
   | `TAVILY_API_KEY` | **Yes (Stage 3+)** | Your Tavily API key — get one at [app.tavily.com](https://app.tavily.com) |
   | `TAVILY_MAX_RESULTS` | No | Number of search results per query, default: `5` |

The `.env` file is listed in `.gitignore` and will **never** be committed to version control. Always use `.env.example` as the reference template.

---

## Run

### Normal startup (no API calls)

```bash
python run.py
```

Expected output:

```
========================================
       AutoResearch AI v0.1.0
========================================

2026-09-30 03:30:00 | INFO     | app.main | AutoResearch AI is starting up...
2026-09-30 03:30:00 | INFO     | app.main | Version     : 0.1.0
2026-09-30 03:30:00 | INFO     | app.main | Environment : development
2026-09-30 03:30:00 | INFO     | app.main | Log Level   : INFO
Environment : development
Status      : READY

2026-09-30 03:30:00 | INFO     | app.main | Startup complete. Application status: READY
```

---

## Stage 3 — Automated Source Discovery

Tavily search and webpage collection are now integrated.

- **Search API**: `app/search/tavily_client.py` wraps the Tavily API and returns typed `Source` objects.
- **Collector**: `app/search/collector.py` fetches URLs, strips boilerplate HTML (nav, footer, scripts), and returns clean `Document` objects.
- **Models**: `app/search/models.py` defines `Source` and `Document` dataclasses.
- **Gemini remains the LLM layer** — the search layer never calls Gemini.
- All Stage 3 tests are fully mocked (no real API calls during `pytest`).

### Configuration

Add to your `.env`:

```env
TAVILY_API_KEY=tvly-your-key-here
TAVILY_MAX_RESULTS=5
```

Get a free Tavily API key at [app.tavily.com](https://app.tavily.com).

### Manual search test

```bash
python run.py --test-search
```

Expected output:

```
========================================
       AutoResearch AI v0.1.0
========================================

Testing Tavily search...
Query: latest developments in artificial intelligence

Found 5 result(s):

  [1] Title of result 1
       https://example.com/article-1  Score: 0.95
  ...

Fetching first result: https://example.com/article-1

Title   : Article Title
URL     : https://example.com/article-1
Length  : 4,321 characters
Preview :
[first 500 chars of extracted text...]

Search pipeline: SUCCESS
```

> **Note:** `python run.py` and `python run.py --test-llm` never call the search API.

The Google Gemini API client is now integrated.

- API keys are loaded securely through `.env` — never hardcoded.
- The Gemini client is isolated in `app/llm/client.py`. The rest of the app imports `GeminiClient` without knowing SDK details.
- Uses the **official `google-genai` SDK** (not the deprecated `google-generativeai`).
- Default model: `gemini-2.5-flash-lite` — fast, capable, and available on the **free tier**.
- Tests in `tests/test_llm.py` mock all API calls — no tokens consumed during testing.

### Manual API connection test

Once `GEMINI_API_KEY` is set in your `.env`, run:

```bash
python run.py --test-llm
```

Expected output:

```
========================================
       AutoResearch AI v0.1.0
========================================

Testing Gemini connection...

Prompt:
Explain Retrieval-Augmented Generation in one sentence.

Response:
[model response here]

Gemini connection: SUCCESS
```

> **Note:** `python run.py` (without `--test-llm`) **never** calls the API.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

All tests are mocked — no API key required to run the test suite.

---

## Stage 5 — Research Agent + Report Generation

The full automated research pipeline is now operational. Given a topic, AutoResearch AI:

1. **Searches** the web via Tavily for relevant sources.
2. **Collects** full webpage content and cleans the HTML.
3. **Chunks** documents into overlapping segments.
4. **Embeds** chunks locally with `all-MiniLM-L6-v2` (no API cost).
5. **Stores** vectors in a local ChromaDB collection.
6. **Retrieves** the most semantically relevant chunks for the topic.
7. **Generates** a structured, grounded report via Google Gemini.
8. **Saves** the report as a Markdown file under `reports/`.

### Architecture

```
Topic
  ↓
Tavily Search API       → Source URLs
  ↓
SourceCollector         → Clean Documents
  ↓
DocumentChunker         → Text Chunks (1000 chars, 150 overlap)
  ↓
EmbeddingService        → Float Vectors (local, free)
  ↓
ChromaDB VectorStore    → Persistent Knowledge Base
  ↓
Semantic Retrieval      → Top-K Relevant Chunks
  ↓
Gemini Prompt           → Grounded, cited prompt with [S1][S2]…
  ↓
Gemini API              → Structured JSON response
  ↓
ResearchReport          → Validated, source-trusted report
  ↓
reports/<topic>.md      → Markdown file saved to disk
```

### New modules

| Module | Purpose |
|---|---|
| `app/research/agent.py` | `ResearchAgent` — orchestrates the full pipeline |
| `app/research/prompts.py` | Prompt builder + source identifier registry |
| `app/research/report.py` | `ResearchReport` + `SourceReference` dataclasses |

### Citation system

Every retrieved chunk is labelled **[S1]**, **[S2]**, etc. before being sent to Gemini.
Gemini is instructed to cite these identifiers for every factual claim.
Source URLs are taken exclusively from Tavily/Document metadata — Gemini **cannot invent URLs**.

### Run the research agent

```bash
python run.py --research "latest developments in artificial intelligence"
```

**Example output:**

```
========================================
  AutoResearch AI — Research Agent
========================================

Topic:
latest developments in artificial intelligence

Searching sources...
Collected sources: 5
Retrieved chunks : 5

Generating report with Gemini...

Research completed successfully.

Executive Summary:
Recent advancements in AI have been remarkable...

Key Findings:
  • LLMs have demonstrated significant improvements… [S1]
  • Generative AI tools are being rapidly deployed… [S2]

Sources:
  [S1] IBM AI Research
       https://www.ibm.com/…
  [S2] …

Report saved to:
reports/latest-developments-in-artificial-intelligence.md
```

### Saved reports

All reports are saved under `reports/` as Markdown files.
Filenames are auto-generated from the topic (lowercased, hyphenated).
The `reports/` directory is **gitignored** — reports are local only.

```
reports/
  latest-developments-in-artificial-intelligence.md
  future-of-quantum-computing.md
  …
```

### Report structure

Each saved Markdown report contains:

```markdown
# Research Report: <topic>

Generated: <timestamp>

## Executive Summary
## Key Findings
## Detailed Analysis
## Emerging Trends
## Conclusion
## Sources
  [S1] Title
  URL
```

> **Note:** Gemini generation is not called by `--test-llm`, `--test-search`, or `--test-rag`.
> Only `--research` triggers the full pipeline.

---

## Stage 6 — Persistent Research Memory & "What's New?"

Stage 6 enables AutoResearch AI to remember past research runs in a local SQLite database (`data/research.db`) and compare current research against prior runs to detect new developments.

### Key Features
- **SQLite Database**: Automatically initializes `data/research.db` storing runs, sources, and findings.
- **Topic Normalization**: Case-insensitive and whitespace-normalized topic lookups.
- **Deterministic Comparator**: Compares new findings and canonical source URLs against previous research without using LLM calls.
- **What's New Reports**: Saves reports to `reports/<topic>-whats-new.md` highlighting New Findings, Changed Findings, New Sources, and Already Known findings.

### Stage 6 Commands

```bash
# Test SQLite database repository and comparator (no network/API calls)
python run.py --test-memory

# View previous research history for a topic
python run.py --history "latest developments in artificial intelligence"

# Run a fresh research cycle and compare it against the previous run
python run.py --whats-new "latest developments in artificial intelligence"
```

---

## Web Application & API (Stage 8)

AutoResearch AI features a FastAPI backend service and a Streamlit interactive web interface.

### 1. Start the FastAPI Backend

```bash
python run.py --api
```

- Server runs at: `http://127.0.0.1:8000`
- Interactive OpenAPI documentation available at: `http://127.0.0.1:8000/docs`
- Health check endpoint: `GET http://127.0.0.1:8000/health`

### 2. Start the Streamlit User Interface

In a second terminal window (with virtual environment active):

```bash
streamlit run app/ui/streamlit_app.py
```

### 3. Usage Steps

1. Start FastAPI backend (`python run.py --api`).
2. Start Streamlit UI (`streamlit run app/ui/streamlit_app.py`).
3. Open the Streamlit URL (usually `http://localhost:8501`) in your browser.
4. Navigate via the sidebar:
   - **Research**: Enter a topic (e.g. "Latest developments in artificial intelligence") and click **Research** to run agent synthesis.
   - **History**: Enter a topic to view all past research runs saved in SQLite.
   - **What's New**: Run a comparison between fresh web research and prior baseline findings.
   - **Schedules**: View, create (daily/weekly), run, enable, disable, and delete automated research schedules.

---

## Development Roadmap

| Stage | Description | Status |
|---|---|---|
| **Stage 1** | Project Foundation | ✅ Complete |
| **Stage 2** | LLM / API Integration (Gemini) | ✅ Complete |
| **Stage 3** | Automated Source Discovery | ✅ Complete |
| **Stage 4** | Document Processing + RAG | ✅ Complete |
| **Stage 5** | Research Agent + Report Generation | ✅ Complete |
| **Stage 6** | Persistent Memory + What's New | ✅ Complete |
| **Stage 7** | Scheduling + Automation | ✅ Complete |
| **Stage 8** | FastAPI + Streamlit UI | ✅ Complete |
| **Stage 9** | Testing + GitHub + Deployment | 🔜 Planned |

---

## Contributing

This project is under active development. Each stage introduces new capabilities. Contributions and feedback are welcome once the core stages are implemented.

---

## License

MIT License — see `LICENSE` for details.

# RAG Evaluation Project

A hands-on project to build a small **Retrieval-Augmented Generation (RAG)** application and, more importantly, a complete **evaluation framework** around it: golden datasets, LLM-as-a-judge metrics, probabilistic acceptance criteria, judge calibration against human review, and automated regression testing in CI.

The knowledge base is the **151 Generation 1 Pokémon**, pulled from [PokéAPI](https://pokeapi.co). Everything runs **locally and for free** using [Ollama](https://ollama.com).

> 🚧 **Work in progress.** This README is updated as each step is completed. See the [Roadmap](#roadmap) for current status.

---

## Table of Contents

- [Why this project](#why-this-project)
- [What is RAG?](#what-is-rag)
- [Why Pokémon?](#why-pokémon)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Step-by-step build log](#step-by-step-build-log)
- [Roadmap](#roadmap)

---

## Why this project

Testing traditional software is deterministic: the same input always produces the same output, so a test either passes or fails. LLM-based systems are **probabilistic**: the same question can produce different answers, and "correct" is often a matter of degree.

This project explores how to bring quality engineering practices to AI systems:

- **Evaluation frameworks** for LLM applications
- **Acceptance criteria** for non-deterministic systems (thresholds instead of exact matches)
- **Regression testing** of prompts and models
- **Calibration** of automated judges against expert (human) review

---

## What is RAG?

**Retrieval-Augmented Generation** is a pattern that lets an LLM answer questions about information it was not trained on.

1. **Indexing (once):** documents are split into chunks and converted into *embeddings* (numeric vectors that capture meaning), so they can be searched semantically.
2. **Retrieval:** when a user asks a question, the system finds the most relevant chunks.
3. **Generation:** the retrieved chunks are sent to the LLM along with the question, and the LLM answers based on them.

A RAG system can fail in two independent places, which makes it a great target for evaluation:

| Failure point | Example | Metric that catches it |
|---|---|---|
| Retrieval | The search returns the wrong Pokémon's data | Context recall / precision |
| Generation | The LLM ignores the context or makes things up | Faithfulness |

---

## Why Pokémon?

- **Verifiable answers:** types, stats, and abilities are exact facts, so correctness is easy to judge.
- **Open data:** PokéAPI is free and well documented.
- **An interesting challenge:** LLMs already know a lot about Pokémon from training. This makes it possible to test whether the model actually answers **from the provided documents** or **from memory**, which is exactly what faithfulness metrics are designed to detect.

---

## Tech stack

| Component | Tool | Role |
|---|---|---|
| Language | Python 3.12 | Everything |
| Local LLM runtime | Ollama | Runs models locally, no API costs |
| Generator model | `llama3.2` (3B) | Answers user questions in the RAG app |
| Judge model | `qwen2.5:7b` | Scores answers (LLM-as-a-judge) |
| Embeddings | `sentence-transformers` | Semantic search over documents |
| Testing | `pytest` | Test runner for evaluations |
| Data source | PokéAPI via `httpx` | Pokémon data |

A different (and larger) model is used as the judge than as the generator, because models tend to be lenient when grading their own output.

---

## Project structure

```
rag-eval-project/
├── data/
│   └── docs/              # 151 Pokémon documents (one .txt per Pokémon)
├── src/
│   ├── __init__.py
│   └── fetch_data.py      # Downloads Pokémon data from PokéAPI
├── evals/                 # Golden dataset, metrics and evaluation tests
├── test_connection.py     # Sanity check that the local LLM responds
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting started

### Prerequisites

- macOS (tested on Apple Silicon M4, 16 GB RAM), Linux should work too
- [Homebrew](https://brew.sh)
- Python 3.10+ (3.12 recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/zeketests/rag-eval-project.git
cd rag-eval-project

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install and start Ollama, then download the models
brew install ollama
brew services start ollama
ollama pull llama3.2
ollama pull qwen2.5:7b

# 5. Verify the LLM connection
python test_connection.py

# 6. Download the Pokémon data
python src/fetch_data.py
```

---

## Step-by-step build log

### Step 1 — Project setup

Created the project folder, initialized a Git repository, and set up a Python **virtual environment** (`.venv`). A virtual environment isolates the project's dependencies from the rest of the system, so package versions don't conflict with other projects.

- `.env` (for secrets such as API keys) and `.venv/` are listed in `.gitignore` so they are never committed.
- Python was upgraded from 3.9 to **3.12**, since modern evaluation libraries (DeepEval, Ragas) require 3.10+.

### Step 2 — Choosing a local LLM with Ollama

The project initially targeted a paid LLM API. To keep it **free and reproducible**, it was switched to **Ollama**, which runs open-source models directly on the local machine:

- No API keys or usage costs
- No rate limits
- Works offline

The trade-off is that small local models are less capable than frontier models, especially as judges. That limitation is intentional here: the calibration step will measure how reliable the judge is and improve it.

### Step 3 — Connecting Python to the LLM

`test_connection.py` sends a simple prompt to `llama3.2` through the `ollama` Python library and prints the answer. This confirms the whole chain works before building anything on top of it.

### Step 4 — Project structure

Organized the code into three folders with clear responsibilities:

- `data/docs/` holds the knowledge base the RAG app will search
- `src/` holds the application code
- `evals/` holds everything related to evaluation

Keeping the application and its evaluations separate mirrors how AI QA is done in real teams.

### Step 5 — Building the knowledge base

`src/fetch_data.py` calls two PokéAPI endpoints for each of the 151 Gen 1 Pokémon:

- `/pokemon/{id}` for name, types, height, weight, abilities, and base stats
- `/pokemon-species/{id}` for the English Pokédex description

Each Pokémon is saved as a plain-text file (for example `001_bulbasaur.txt`):

```
Name: Bulbasaur
Pokedex number: 1
Types: grass, poison
Height: 0.7 m
Weight: 6.9 kg
Abilities: overgrow, chlorophyll (hidden)
Base stats: hp 45, attack 49, defense 49, special-attack 65, special-defense 65, speed 45
Description: ...
```

Implementation notes:

- PokéAPI returns height in decimeters and weight in hectograms, so both are divided by 10 to get meters and kilograms.
- A short delay (`time.sleep(0.2)`) between requests avoids overloading the free API.

### Step 6 — Version control

Pushed the project to GitHub with the default branch renamed to `main`, verifying with `git status` that no secrets or virtual environment files were included.

### Step 7 — Dependency tracking

Added `requirements.txt` so anyone (including the CI pipeline) can reproduce the environment with a single command.

---

## Roadmap

- [x] Project setup and local LLM
- [x] Knowledge base from PokéAPI
- [ ] **RAG app:** chunking, embeddings, retrieval, and answer generation
- [ ] **Golden dataset:** 50–100 test questions with expected answers, including edge cases and unanswerable questions
- [ ] **Metrics:** faithfulness, answer relevancy, and context recall using an LLM-as-a-judge
- [ ] **Acceptance criteria:** threshold-based pass/fail for a probabilistic system, with repeated runs to measure variance
- [ ] **Judge calibration:** compare judge scores against manual human scores (Cohen's kappa / correlation) and refine the judge rubric
- [ ] **Regression testing:** compare scores across prompt and model versions
- [ ] **CI/CD:** run evaluations automatically with GitHub Actions
- [ ] **Results and lessons learned**

---

## Data attribution

Pokémon data is provided by [PokéAPI](https://pokeapi.co). Pokémon and Pokémon character names are trademarks of Nintendo. This is a non-commercial educational project.
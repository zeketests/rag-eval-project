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
- [Current results](#current-results)
- [Key lessons](#key-lessons)
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
│   ├── fetch_data.py      # Downloads Pokémon data from PokéAPI
│   ├── retriever.py       # Embedding-based semantic search
│   └── rag.py             # Full RAG pipeline: retrieve + generate
├── evals/
│   ├── generate_dataset.py  # Builds auto-generated test cases from the docs
│   ├── golden_dataset.json  # 40 auto-generated single-fact cases
│   ├── manual_cases.json    # Hand-written edge cases
│   ├── judge.py             # LLM-as-a-judge with rule-based pre-checks
│   ├── run_eval.py          # Runs the RAG on every case and reports metrics
│   └── results/             # Timestamped results of each evaluation run
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

# 7. Try the RAG app
python -m src.rag

# 8. Generate the dataset and run the full evaluation
python evals/generate_dataset.py
python -m evals.run_eval
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

### Step 8 — Retriever (semantic search)

`src/retriever.py` converts every document and every question into an **embedding** using the `all-MiniLM-L6-v2` model, then returns the documents whose vectors are closest to the question (cosine similarity).

Since each Pokémon file is short, **each document is a single chunk**, so no splitting is needed.

Observation: for "What type is Pikachu?", the top result was correct (Pikachu), but the 2nd and 3rd were Pidgeot and Pidgeotto, likely because of the similar "Pi-" prefix. Embedding similarity is not always semantically meaningful, which is why retrieval needs to be measured, not assumed.

### Step 9 — Generation (completing the RAG)

`src/rag.py` retrieves the top 3 documents and sends them to `llama3.2` with a prompt that instructs the model to:

- answer **only** using the provided context
- reply exactly `"I don't know."` when the context does not contain the answer

`temperature` is set to `0` to reduce output variability, which makes evaluation more stable. The pipeline returns the retrieved contexts along with the answer, because evaluation metrics need them.

### Step 10 — Golden dataset (auto-generated)

`evals/generate_dataset.py` builds 40 test cases by picking random Pokémon and filling question templates ("What type is {name}?", "How much does {name} weigh?", etc.). The expected answer is read directly from the source document, and the source file name is stored so retrieval can be checked too.

A **fixed random seed** guarantees the same dataset on every run. This is essential for regression testing: you cannot compare two runs if the test set changes between them.

### Step 11 — Hand-written edge cases

Auto-generated cases are all the same shape: one clean, well-phrased question about one Pokémon. Real users are messier, so `evals/manual_cases.json` adds cases that target specific risks:

| Category | What it tests |
|---|---|
| `comparison` | Retrieval of **two** documents and reasoning across them |
| `multi_fact` | Combining several facts from one document |
| `typo` / `paraphrase` | Robustness to misspellings and different phrasings |
| `description_lookup` | Finding a Pokémon by its description instead of its name |
| `unanswerable` | A Pokémon that is not in the knowledge base (Gen 2) |
| `memory_trap` | Questions the model **knows** from training but the documents don't cover (e.g. evolutions). Answering them means the model ignored the context. |

Every expected answer was verified against the source documents. For example, a case asking for "a red lizard with big eyes" was classified as a memory trap, because Charmander's document does not contain that description.

### Step 12 — Evaluation runner

`evals/run_eval.py` runs the RAG on every case, checks whether the expected documents were retrieved (**retrieval hit rate**), and saves timestamped results to `evals/results/` so runs can be compared over time.

### Step 13 — LLM-as-a-judge and its calibration

`evals/judge.py` uses `qwen2.5:7b` to label each answer as `correct`, `partial`, or `incorrect` against the expected answer.

Before trusting the judge, it was tested on 5 cases with obvious expected labels. The first version agreed on only **2/5**, including a critical bug: it scored an **over-refusal** ("I don't know." when an answer existed) as a perfect answer. Three design changes brought it to **5/5**:

1. **Rule-based checks for refusals.** "I don't know." cases are decided by code, not by the LLM. A small model struggles with conditional rules, and a deterministic check is always right. Principle: *don't ask an LLM to do what code can do with certainty.*
2. **A 3-label scale instead of 1–5.** The difference between a 4 and a 5 is subjective even for humans; a small judge cannot apply it consistently.
3. **Few-shot examples** in the judge prompt. The examples use different Pokémon than the test cases, to avoid the judge passing because it has already seen the answers.

### Step 14 — Error analysis

The first full run reported **89% correctness**. Every failure was then reviewed by hand and classified by root cause:

| Root cause | Cases | What happened |
|---|---|---|
| **Dataset error** | 5 | Expected answers included details the question never asked for, such as the "(hidden)" ability tag or the numbers behind a comparison |
| **Judge error** | 2 | "Fighting." was marked wrong against "fighting" because of capitalization and punctuation |
| **RAG error** | 1 | The model quoted both weights correctly (Snorlax 460 kg, Onix 210 kg) but concluded that Onix was heavier |

Fixes:

- Removed the "(hidden)" tag from generated expected answers and simplified comparison answers.
- Added an **exact-match check after normalization** (lowercase, trimmed, no trailing period) before calling the judge. A "contains" check was deliberately avoided: the wrong Snorlax/Onix answer *mentions* Snorlax, so "contains" would have hidden a real failure.

After the fixes, correctness rose to **98%**, and the only remaining failure is the genuine RAG error.

---

## Current results

Latest run: 50 test cases (40 auto-generated + 10 hand-written).

| Category | Retrieval hit rate | Correctness | n |
|---|---|---|---|
| single_fact | 40/40 | 100% | 40 |
| comparison | 2/2 | 50% | 2 |
| description_lookup | 1/1 | 100% | 1 |
| multi_fact | 1/1 | 100% | 1 |
| paraphrase | 1/1 | 100% | 1 |
| typo | 1/1 | 100% | 1 |
| memory_trap | n/a | 100% | 3 |
| unanswerable | n/a | 100% | 1 |
| **Overall** | | **98%** | 50 |

Correctness score: `correct` = 1, `partial` = 0.5, `incorrect` = 0.

**Findings:**

- **Retrieval is not the bottleneck.** Every expected document was retrieved.
- **Cross-document reasoning is the main weakness.** In the failed comparison, retrieval was perfect but the 3B generator drew the wrong conclusion from correct data.
- **The model respects the context.** All memory traps were answered with "I don't know." instead of facts from training data.

**Known limitations:**

- Edge-case categories have only 1–3 cases each, which is too few for statistically meaningful conclusions. A target of at least 10 cases per category is planned.
- The judge was validated on only 5 cases. A larger calibration against human labels is planned.
- Each case was run once. Variance across repeated runs has not been measured yet.

---

## Key lessons

1. **A metric is only as reliable as the dataset and the judge behind it.** The first run reported 89%, but only 1 of 8 failures was a real system error. Without error analysis, that number would have been misleading.
2. **Always validate the judge before using it.** An LLM judge is itself a probabilistic system and can be wrong in systematic ways.
3. **Prefer deterministic checks where possible.** Rules handle refusals and exact matches reliably; the LLM judge is reserved for cases that need interpretation.
4. **Design rules carefully.** A naive "contains" check would have turned a real failure into a pass.
5. **Separate retrieval from generation.** Measuring both independently showed that the comparison failure came from reasoning, not from search.

---

## Roadmap

- [x] Project setup and local LLM
- [x] Knowledge base from PokéAPI
- [x] **RAG app:** embeddings, retrieval, and answer generation
- [x] **Golden dataset:** 50 test cases, including edge cases and unanswerable questions
- [x] **Retrieval metric:** retrieval hit rate per category
- [x] **Correctness metric:** LLM-as-a-judge with rule-based pre-checks, validated on known cases
- [x] **Error analysis:** failures classified as dataset, judge, or system errors
- [ ] **More edge cases:** at least 10 cases per category
- [ ] **More metrics:** faithfulness and answer relevancy
- [ ] **Acceptance criteria:** threshold-based pass/fail for a probabilistic system, with repeated runs to measure variance
- [ ] **Judge calibration:** compare judge scores against manual human scores (Cohen's kappa / correlation) and refine the judge rubric
- [ ] **Regression testing:** compare scores across prompt and model versions
- [ ] **CI/CD:** run evaluations automatically with GitHub Actions
- [ ] **Results and lessons learned**

---

## Data attribution

Pokémon data is provided by [PokéAPI](https://pokeapi.co). Pokémon and Pokémon character names are trademarks of Nintendo. This is a non-commercial educational project.
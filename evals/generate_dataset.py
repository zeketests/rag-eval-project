import json
import random
from pathlib import Path

DOCS_DIR = Path("data/docs")
OUT_FILE = Path("evals/golden_dataset.json")
N_CASES = 40
random.seed(42)  # fixed seed so the dataset is reproducible


def parse_doc(path: Path) -> dict:
    """Turn a 'Key: value' text file into a dict."""
    fields = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(": ")
        fields[key] = value
    return fields


# Question templates: (question, field holding the expected answer)
TEMPLATES = [
    ("What type is {name}?", "Types"),
    ("How much does {name} weigh?", "Weight"),
    ("How tall is {name}?", "Height"),
    ("What abilities does {name} have?", "Abilities"),
    ("What is {name}'s Pokedex number?", "Pokedex number"),
]

docs = sorted(DOCS_DIR.glob("*.txt"))
cases = []

for i, path in enumerate(random.sample(docs, N_CASES), start=1):
    poke = parse_doc(path)
    question, field = random.choice(TEMPLATES)
    cases.append({
        "id": f"auto_{i:03d}",
        "category": "single_fact",
        "question": question.format(name=poke["Name"]),
        "expected_answer": poke[field].replace(" (hidden)", ""),
        "expected_sources": [path.name],
    })

OUT_FILE.write_text(json.dumps(cases, indent=2))
print(f"Generated {len(cases)} cases in {OUT_FILE}")
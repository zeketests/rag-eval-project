import json
from datetime import datetime
from pathlib import Path

from src.rag import answer
from evals.judge import judge_correctness

DOCS_DIR = Path("data/docs")
RESULTS_DIR = Path("evals/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def load_cases() -> list[dict]:
    cases = []
    for file in ["evals/golden_dataset.json", "evals/manual_cases.json"]:
        cases += json.loads(Path(file).read_text())
    return cases


def retrieval_hit(case: dict, contexts: list[str]) -> bool | None:
    """True if every expected document was retrieved. None if nothing was expected."""
    if not case["expected_sources"]:
        return None
    expected_texts = [(DOCS_DIR / s).read_text() for s in case["expected_sources"]]
    return all(text in contexts for text in expected_texts)


results = []
for case in load_cases():
    output = answer(case["question"])
    hit = retrieval_hit(case, output["contexts"])
    verdict = judge_correctness(case["question"], case["expected_answer"], output["answer"])
    results.append({
        **case,
        "answer": output["answer"],
        "retrieval_hit": hit,
        "label": verdict["label"],
        "judge_reason": verdict["reason"],
    })
    print(f"[{case['id']}] {verdict['label']:9s} | Q: {case['question']}")
    print(f"    expected: {case['expected_answer']}")
    print(f"    got:      {output['answer']}\n")

categories = sorted({r["category"] for r in results})

# Retrieval summary per category
print("=== Retrieval hit rate by category ===")
for cat in categories:
    hits = [r["retrieval_hit"] for r in results if r["category"] == cat and r["retrieval_hit"] is not None]
    if hits:
        print(f"{cat:20s} {sum(hits)}/{len(hits)}")

# Correctness summary per category (correct = 1, partial = 0.5, incorrect = 0)
print("\n=== Correctness by category ===")
for cat in categories:
    scores = [LABEL_SCORE[r["label"]] for r in results if r["category"] == cat]
    print(f"{cat:20s} {sum(scores) / len(scores):.0%}  (n={len(scores)})")

overall = sum(LABEL_SCORE[r["label"]] for r in results) / len(results)
print(f"\nOVERALL CORRECTNESS: {overall:.0%}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = RESULTS_DIR / f"run_{timestamp}.json"
out_file.write_text(json.dumps(results, indent=2))
print(f"\nSaved results to {out_file}")
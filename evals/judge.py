import json
import ollama

JUDGE_MODEL = "qwen2.5:7b"

JUDGE_PROMPT = """You are a strict evaluator of a question-answering system.
Compare the ACTUAL answer with the EXPECTED answer and assign a label.

Labels:
- "correct": contains ALL the expected information. Extra correct details are fine.
- "partial": contains some of the expected information but misses part of it.
- "incorrect": wrong, irrelevant, or contradicts the expected answer.

Examples:
Q: What type is Charmander? | EXPECTED: fire | ACTUAL: Charmander is a Fire-type Pokemon.
-> {{"label": "correct", "reason": "States the correct type."}}

Q: What are Onix's types? | EXPECTED: rock, ground | ACTUAL: Rock.
-> {{"label": "partial", "reason": "Missing the ground type."}}

Q: How much does Snorlax weigh? | EXPECTED: 460.0 kg | ACTUAL: 120 kg.
-> {{"label": "incorrect", "reason": "Wrong weight."}}

Now evaluate:
Q: {question} | EXPECTED: {expected} | ACTUAL: {actual}

Reply ONLY with JSON: {{"label": "<correct|partial|incorrect>", "reason": "<one short sentence>"}}"""


def is_refusal(text: str) -> bool:
    """Detect answers that decline to respond."""
    return "i don't know" in text.lower()

def normalize(text: str) -> str:
    """Lowercase, trim whitespace and trailing periods for exact comparison."""
    return " ".join(text.lower().strip().rstrip(".").split())



def judge_correctness(question: str, expected: str, actual: str) -> dict:
    """Score an answer from 1 to 5.

    Refusal cases are decided by deterministic rules; everything else
    is scored by the judge model.
    """
    expected_refusal = is_refusal(expected)
    actual_refusal = is_refusal(actual)

    if expected_refusal and actual_refusal:
        return {"label": "correct", "reason": "Correctly declined to answer (rule-based)."}
    if expected_refusal:
        return {"label": "incorrect", "reason": "Answered when it should have declined (rule-based)."}
    if actual_refusal:
        return {"label": "incorrect", "reason": "Declined when an answer was expected (rule-based)."}
    if normalize(expected) == normalize(actual):
        return {"label": "correct", "reason": "Exact match after normalization (rule-based)."}
        
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, expected=expected, actual=actual
        )}],
        format="json",
        options={"temperature": 0},
    )
    return json.loads(response["message"]["content"])


if __name__ == "__main__":
    tests = [
        ("What type is Squirtle?", "water", "Squirtle is a Water type.", "correct"),
        ("What type is Squirtle?", "water", "Fire.", "incorrect"),
        ("What type is Squirtle?", "water", "I don't know.", "incorrect"),
        ("What are Mewtwo's types and base HP?", "psychic, base HP 106", "Psychic.", "partial"),
        ("What does Pikachu evolve into?", "I don't know.", "Raichu", "incorrect"),
    ]
    correct = 0
    for q, expected, actual, want in tests:
        result = judge_correctness(q, expected, actual)
        ok = result["label"] == want
        correct += ok
        print(f"{'OK  ' if ok else 'FAIL'} want={want} got={result['label']} <- {actual}")
    print(f"\nJudge agreement: {correct}/{len(tests)}")
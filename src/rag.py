import ollama
from src.retriever import retrieve

GENERATOR_MODEL = "llama3.2"

PROMPT_TEMPLATE = """You are a Pokemon expert assistant.
Answer the question using ONLY the information in the context below.
If the context does not contain the answer, reply exactly: "I don't know."

Context:
{context}

Question: {question}
Answer:"""


def answer(question: str, k: int = 3) -> dict:
    """Run the full RAG pipeline: retrieve context, then generate an answer."""
    contexts = retrieve(question, k=k)
    prompt = PROMPT_TEMPLATE.format(
        context="\n---\n".join(contexts), question=question
    )
    response = ollama.chat(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return {
        "question": question,
        "answer": response["message"]["content"].strip(),
        "contexts": contexts,
    }


if __name__ == "__main__":
    for q in [
        "What type is Pikachu?",
        "How much does Snorlax weigh?",
        "Who is the trainer of Ash's Pikachu?",
    ]:
        result = answer(q)
        print(f"Q: {q}\nA: {result['answer']}\n")
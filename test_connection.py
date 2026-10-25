import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Decí hola en una frase corta."}],
)

print(response["message"]["content"])
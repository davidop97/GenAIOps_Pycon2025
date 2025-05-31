# ask.py
import os
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

def main():
    # 1) Carga el índice FAISS ya generado
    vectordb = load_vectorstore_from_disk()

    # 2) Construye el pipeline RAG con tu prompt actual
    chain = build_chain(vectordb, prompt_version=os.getenv("PROMPT_VERSION"))

    # 3) Lee una pregunta desde la consola
    pregunta = input("¿En qué puedo ayudarte sobre AWS Data Pipelines? ")

    # 4) Ejecuta la consulta y muestra la respuesta
    resultado = chain({"question": pregunta, "chat_history": []})
    print("\n➡️ Respuesta:\n", resultado["answer"])

if __name__ == "__main__":
    main()

from langchain_chroma import Chroma
import shutil
import os

# Fonction pour créer et configurer le vector store Chroma
def save_to_chroma(embeddings, chunks, persist_directory="data/chroma_db/"):
    # Nettoyage de l'ancienne base si elle existe
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store
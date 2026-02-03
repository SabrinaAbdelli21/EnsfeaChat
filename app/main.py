import os
from rag.parsing import get_loader
from rag.chunking import chunk_documents
from rag.embeddings import get_embedding_model, embed_documents
from rag.vector_store import save_to_chroma
from rag.generator import generate_answer
from rag.retriever import retrieve_context

# Programme principale 
if __name__ == "__main__":

    file_path = "data/corpus/"

    # Parsing des documents
    try:
        documents = get_loader(file_path)
    except Exception as e:
        print(f"Erreur lors du chargement des documents: {e}")

    # Chunking des documents
    try:
        chunked_docs = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
    except Exception as e:
        print(f"Erreur lors du chunking des documents: {e}")

    # génération des embeddings
    try:    
        embed_model = get_embedding_model()
        vectors = embed_documents(chunked_docs, embed_model)
    except Exception as e:
        print(f"Erreur lors de la génération des embeddings: {e}")

    # Sauvegarde dans le vector store Chroma
    try:
  
        vector_store = save_to_chroma(embed_model, chunked_docs)

        # Test 
        query = "Quel est le numéro de carte de transport ?"
            
    
        # La réponse la plus proche de la requete
        relevant_chunks = retrieve_context(query, vector_store, top_k=5)
        print(f"{len(relevant_chunks)} chunks pertinents trouvés")

    except Exception as e:
        print(f"Erreur lors de la sauvegarde dans le vector store Chroma: {e}")
    
    # Génération de réponse avec LLM
    try:
        answer = generate_answer(query, relevant_chunks)
        print("\n--- Réponse Générée par le modèle ---")
        print(answer)

    except Exception as e:
        print(f"Erreur lors de la génération de la réponse: {e}")
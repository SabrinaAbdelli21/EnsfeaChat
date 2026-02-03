import os
from rag.parsing import get_loader
from rag.chunking import chunk_documents
from rag.embeddings import get_embedding_model, embed_documents
from rag.vector_store import save_to_chroma
from rag.generator import generate_answer
from rag.retriever import retrieve_context

def run_rag(question, data_path):
    

    # Parsing des documents
    documents = get_loader(data_path)

    # Chunking des documents
    chunked_docs = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

    # génération des embeddings
    embed_model = get_embedding_model()

    # Sauvegarde dans le vector store Chroma
    vector_store = save_to_chroma(embed_model, chunked_docs)

    # Test 
    query = "Quel est le numéro de carte de transport ?"
            
    # La réponse la plus proche de la requete
    relevant_chunks = retrieve_context(query, vector_store, top_k=5)
    
    # Génération de réponse avec LLM
    answer = generate_answer(query, relevant_chunks)


    return answer
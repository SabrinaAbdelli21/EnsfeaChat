# Programme principale 
import os
from rag.parsing import get_loader
from rag.chunking import chunk_documents
from rag.embeddings import get_embedding_model, embed_documents

if __name__ == "__main__":

    file_path = "data/corpus/"

    # Parsing des documents
    try:
        documents = get_loader(file_path)
        print(f"Nombre de documents chargés: {len(documents)}")
    except Exception as e:
        print(f"Erreur lors du chargement des documents: {e}")

    # Chunking des documents
    try:
        chunked_docs = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

        print(f"Nombre de chunks créés: {len(chunked_docs)}")
        print("Nombre de chunks par document:") 

        doc_chunk_counts = {}
        for doc in chunked_docs:  
            doc_id = doc.metadata.get("source", "unknown")
            if doc_id not in doc_chunk_counts:
                doc_chunk_counts[doc_id] = 0
            doc_chunk_counts[doc_id] += 1

        for doc_id, count in doc_chunk_counts.items():
            print(f"{doc_id}: {count} chunks")

        print("Premiers chunk:")
        print(chunked_docs[0].page_content)
        print("Metadata du premier chunk:")
        print(chunked_docs[0].metadata)

    except Exception as e:
        print(f"Erreur lors du chunking des documents: {e}")

    # génération des embeddings
    try:    
        embed_model = get_embedding_model()
        vectors = embed_documents(chunked_docs, embed_model)
        
        # 3. Vérification
        print(f"Dimension d'un vecteur : {len(vectors[0])}")
        print(f"Exemple des 5 premières valeurs du premier vecteur : {vectors[0][:5]}")

    except Exception as e:
        print(f"Erreur lors de la génération des embeddings: {e}")
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Fonction pour confogurer et charger le modèle d'embeddings
def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    # On définit les paramètres pour que le modèle tourne en local sur CPU
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} 
    
    print(f"Loading embedding model: {model_name}...")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="/app/models_cache" 
    )

# Fonction pour embedder les chunks de documents
def embed_documents(chunks, model):
    
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.embed_documents(texts)
    
    print(f"{len(embeddings)} vecteurs générés avec succès.")
    return embeddings


# Fonction pour récupérer les chunks pertinents en fonction de la question
def retrieve_context(question, vector_store, top_k = 5 ) :
    relevant_chunks = vector_store.similarity_search(question, k=top_k)
    return relevant_chunks
from rag.pipeline.pipeline import run_rag

# Programme principale 
if __name__ == "__main__":

    file_path = "data/corpus/"
    query = "Quel est le numéro de carte de transport ?"

    # Exécution du pipeline RAG
    answer = run_rag(query, file_path)
    print("\n La requete : ", query)
    print("--- Réponse Générée par le modèle ---")
    print(answer)
    print("-------------------------------------")
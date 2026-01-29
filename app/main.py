# Programme principale 
import os
from rag.parsing import get_loader


if __name__ == "__main__":

    file_path = "data/corpus/"

    # Parsing des documents
    try:
        documents = get_loader(file_path)
        print(f"Nombre de documents chargés: {len(documents)}")
        
        for i, doc in enumerate(documents):
            # On récupère le nom du fichier dans les métadonnées
            source = doc.metadata.get('source', 'Inconnue')
            
            print(f"\n--- DOCUMENT {i+1} ({source}) ---")
            # .strip() enlève les espaces vides inutiles au début/fin
            print(doc.page_content[:300].strip()) 
            print("-" * 40)
            
    except Exception as e:
        print(f"Erreur lors du chargement des documents: {e}")
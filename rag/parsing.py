# Importation des bibliotheques
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    DirectoryLoader 
)
from langchain_core.documents import Document 
from pathlib import Path
from typing import List, Union


# Fonction de parsing des documents
def get_loader(file_path: Union[str, Path], **kwargs) -> List[Document]:
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier ou répertoire {file_path} n'existe pas")
    
    # Gestion des répertoires
    if file_path.is_dir():
        # Scanne de tout le dossier
        return _load_directory(file_path, **kwargs)
    
    # Gestion des fichiers individuels
    file_extension = file_path.suffix.lower()
    
    if file_extension == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8") # encodage utf-8
    elif file_extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_extension == ".docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        raise ValueError(f"Format de fichier non supporté: {file_extension}")
    
    documents = loader.load()
    return documents


# Fonction pour charger tous les documents du corpus
def _load_directory(directory_path: Path, **kwargs) -> List[Document]:
    all_docs = []
    # On cherche récursivement tous les fichiers
    for p in directory_path.rglob("*"):
        if p.suffix.lower() in [".txt", ".pdf", ".docx"]:
            try:
                # On utilise ta fonction get_loader
                all_docs.extend(get_loader(p))
            except Exception as e:
                print(f"Erreur sur {p.name}: {e}")
    return all_docs


# Découpage des docs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List


# Fonction pour créer un chunker de texte 
def get_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list = None,
) -> RecursiveCharacterTextSplitter:
 
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        add_start_index=True
    )
    
    return text_splitter


# Fonction pour chunker une liste de documents
def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list = None,
) -> list[Document]:

    # Crée le chunker
    text_splitter = get_chunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    
    # Chunk tous les documents
    all_chunks = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = doc.metadata.copy()
            chunk_metadata.update({"chunk_index": i})
            
            chunk_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            all_chunks.append(chunk_doc)
    
    return all_chunks
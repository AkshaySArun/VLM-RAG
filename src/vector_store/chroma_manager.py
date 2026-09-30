import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

class ChromaManager:
    def __init__(self, persist_directory: str = None, collection_name: str = "multimodal_rag", embedding_function: Any = None):
        """
        Initializes the ChromaDB manager using LangChain's wrapper.
        """
        base_persist = persist_directory or os.getenv("VECTOR_DB_PATH", "./data/chroma")
        self.persist_directory = str(Path(base_persist))
        self.collection_name = collection_name
        self.embedding_function = embedding_function # We'll pass our CLIP embedder here later
        
        # Initialize LangChain Chroma client
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function
        )
        print(f"[+] LangChain-Chroma initialized at {self.persist_directory}")

    def add_embeddings(self, 
                       ids: List[str], 
                       embeddings: List[Any], 
                       metadatas: List[Dict[str, Any]], 
                       documents: List[str]):
        """
        Adds embeddings and metadata to the collection.
        """
        if not ids:
            return
            
        try:
            # Ensure embeddings are standard Python floats
            casted_embeddings = []
            for emb in embeddings:
                if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                    emb = emb[0]
                casted_embeddings.append([float(v) for v in emb])
            
            print(f"[*] Adding {len(ids)} items to LangChain-Chroma...", flush=True)
            self.vectorstore._collection.add(
                ids=ids,
                embeddings=casted_embeddings,
                metadatas=metadatas,
                documents=documents
            )
            print(f"[+] Added {len(ids)} items. Total: {self.get_count()}", flush=True)
        except Exception as e:
            print(f"[!] Critical error in ChromaManager.add_embeddings: {e}")

    def query(self, 
              query_embeddings: List[List[float]], 
              n_results: int = 5, 
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Native query support for multimodal embeddings.
        """
        try:
            # LangChain Chroma doesn't have a direct 'query_by_embedding' that returns the same format as raw chroma
            # But we can access the underlying collection
            results = self.vectorstore._collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=["metadatas", "documents", "distances"]
            )
            return results
        except Exception as e:
            print(f"[!] Error querying ChromaDB: {e}")
            return {}

    def get_count(self) -> int:
        return self.vectorstore._collection.count()

    def delete_paper(self, paper_id: str) -> bool:
        """
        Deletes all documents/embeddings associated with a paper_id.
        """
        try:
            self.vectorstore._collection.delete(where={"paper_id": paper_id})
            print(f"[+] Deleted vector store entries for paper_id: {paper_id}")
            return True
        except Exception as e:
            print(f"[!] Error deleting vectors for paper_id {paper_id}: {e}")
            return False

if __name__ == "__main__":
    # Quick sanity check
    manager = ChromaManager()
    print(f"Collection count: {manager.get_count()}")

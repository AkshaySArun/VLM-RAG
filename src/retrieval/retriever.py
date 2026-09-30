from typing import List, Dict, Any, Optional
from src.embeddings.model_loader import MultimodalEmbedder
from src.vector_store.chroma_manager import ChromaManager

class MultimodalRetriever:
    def __init__(self, embedder: MultimodalEmbedder, vector_store: ChromaManager):
        """
        Initializes the retriever with an embedder and a vector store.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, paper_id: Optional[str] = None, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs text-to-multimodal retrieval with optional paper_id document isolation filter.
        """
        print(f"[*] Retrieving context for query: '{query}' (paper_id filter: {paper_id})")
        
        # 1. Encode the text query into the CLIP shared space
        query_embedding = self.embedder.encode_text(query).cpu().detach().tolist()[0]
        
        # 2. Query ChromaDB with paper_id metadata filter
        where_clause = {"paper_id": paper_id} if paper_id else None
        
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        formatted_results = []
        if results and "documents" in results and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if ("metadatas" in results and len(results["metadatas"]) > 0) else [{}] * len(docs)
            distances = results["distances"][0] if ("distances" in results and len(results["distances"]) > 0) else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metas, distances):
                score = round(1.0 - float(dist), 4) if dist is not None else 0.0
                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "score": score
                })
            
        print(f"[+] Retrieved {len(formatted_results)} relevant items for paper_id={paper_id}.")
        return formatted_results

if __name__ == "__main__":
    print("MultimodalRetriever module loaded.")

import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    db_dirs = [d for d in current_dir.iterdir() if d.is_dir() and (d / "chroma.sqlite3").exists()]

    for db_dir in db_dirs:
        try:
            client = chromadb.PersistentClient(path=str(db_dir))
            collections = client.list_collections()
            
            for collection in collections:
                key = f"{db_dir.name}/{collection.name}"
                backends[key] = {
                    "directory": str(db_dir),
                    "collection_name": collection.name,
                    "display_name": f"{db_dir.name} - {collection.name}",
                    "count": str(collection.count())
                }
        except Exception as e:
            backends[f"error_{db_dir.name}"] = {
                "directory": str(db_dir),
                "collection_name": "N/A",
                "display_name": f"ERROR: {db_dir.name} ({str(e)[:30]}...)",
                "count": "0"
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple[Any, bool, str]:
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name=collection_name)
        return collection, True, ""
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    where_filter = None
    if mission_filter and mission_filter.lower() != "all":
        where_filter = {"mission": mission_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    context_parts = ["Retrieved Context:"]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        mission = meta.get("mission", "Unknown Mission").replace("_", " ").title()
        source = meta.get("source", "Unknown Source")
        category = meta.get("document_category", "Unknown Category").replace("_", " ").title()
        
        header = f"[{i}] Mission: {mission} | Source: {source} | Category: {category}"
        context_parts.append(header)
        
        # Truncate if too long
        content = doc[:2000] + "..." if len(doc) > 2000 else doc
        context_parts.append(content)
        context_parts.append("-" * 20)

    return "\n".join(context_parts)
import torch
import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE_DIR = Path(__file__).parent.parent
DB_PATH = str(BASE_DIR / "db")

_client = None
_collection = None


def get_device():
    """Detects hardware accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_collection():
    """Singleton to keep DB and Embedding model in memory."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        embedding_func = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3", device=get_device()
        )
        _collection = _client.get_collection(
            name="f1_2026_regulations", embedding_function=embedding_func
        )
    return _collection


def fetch_full_rule(rule_id: str) -> list:
    """
    Surgical Fetcher: Matches exact IDs across all regulatory sections.
    """
    collection = get_collection()
    clean_id = rule_id.upper().replace("ARTICLE", "").strip()

    prefixes = ["", "A", "B", "C", "D", "E", "F"]
    variants = []
    for p in prefixes:
        if clean_id[0].isalpha() and p != "":
            continue
        variants.append(f"{p}{clean_id}")

    results = collection.get(
        where={"rule_id": {"$in": list(set(variants))}},
        include=["documents", "metadatas"],
    )

    if results and results["documents"]:
        return list(zip(results["documents"], results["metadatas"]))

    return []

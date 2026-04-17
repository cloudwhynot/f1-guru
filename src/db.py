import json
import torch
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def get_device():
    """
    Detects the best available hardware accelerator.
    Logic: CUDA (Nvidia) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_chunks_from_jsonl(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing JSONL file at {file_path}")
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def initialize_vector_db():
    base_dir = Path(__file__).parent.parent
    jsonl_path = base_dir / "data" / "chunked" / "chunks.jsonl"
    db_path = base_dir / "db"

    client = chromadb.PersistentClient(path=str(db_path))
    device = get_device()
    print(f"DATABASE: Initializing BAAI/bge-m3 on device: '{device}'")

    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3", device=device
    )

    collection = client.get_or_create_collection(
        name="f1_2026_regulations",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"DATABASE: Reading chunks from {jsonl_path}...")
    chunks = load_chunks_from_jsonl(jsonl_path)

    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        rule_id = chunk["metadata"].get("rule_id", "unknown")
        ids.append(f"id_{rule_id}_{i}")
        documents.append(chunk["page_content"])
        metadatas.append(chunk["metadata"])

    BATCH_SIZE = 20
    total_chunks = len(documents)
    print(f"DATABASE: Syncing {total_chunks} chunks in batches of {BATCH_SIZE}...")

    for i in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_chunks)

        batch_ids = ids[i:batch_end]
        batch_docs = documents[i:batch_end]
        batch_metas = metadatas[i:batch_end]

        print(f"  > Processing batch {i//BATCH_SIZE + 1} ({i} to {batch_end})...")

        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    print(f"SUCCESS: Database created/updated at {db_path}")
    return collection


def test_retrieval(collection, query: str, filter_metadata: dict = None):
    """
    Advanced retrieval: Supports semantic search and metadata filtering.
    Example filter_metadata: {"article": "Article A1"}
    """
    print(f"\n--- SEARCHING: '{query}' ---")
    if filter_metadata:
        print(f"FILTER: {filter_metadata}")

    results = collection.query(query_texts=[query], n_results=2, where=filter_metadata)

    if not results["documents"][0]:
        print("No results found.")
        return

    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        print(
            f"\n[RANK {i+1}] Rule: {meta.get('rule_id')} | Page: {meta.get('page_number')}"
        )
        print(f"BREADCRUMB: {meta.get('breadcrumb')}")
        print(f"CONTENT: {doc[:300]}...")
    print("-" * 30)


if __name__ == "__main__":
    try:
        f1_collection = initialize_vector_db()

        # Scenario 1: Natural Language Query (General)
        test_retrieval(f1_collection, "What is the minimum weight of the car?")

        # Scenario 2: Technical/Keyword Search
        test_retrieval(f1_collection, "ERS energy recovery limits")

        # Scenario 3: Specific Article Search
        test_retrieval(f1_collection, "Rules defined in Article 4.1")

        # Scenario 4: Metadata Filtering (The most powerful way to search)
        test_retrieval(
            f1_collection, "weight", filter_metadata={"section": "Section A"}
        )

    except Exception as e:
        print(f"FATAL ERROR: {e}")

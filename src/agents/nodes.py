import torch
import chromadb
from pathlib import Path
from ollama import chat
from langgraph.types import Command
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from .state import AgentState

BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = str(BASE_DIR / "db")


def get_device() -> str:
    """
    Detects the best available hardware accelerator.
    Logic: CUDA (Nvidia) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


def retrieve_node(state: AgentState):
    """
    NODE: Retriever
    Fetches chunks from ChromaDB. Includes 'Pivot Boosting' for loops.
    """
    print(f"\n--- NODE: RETRIEVER (Loop: {state['loop_count']}) ---")

    search_term = state.get("transformed_query") or state["query"]

    if state["loop_count"] > 0:
        search_term = f"FIA sanctions penalties breach non-compliance technical regulation {search_term}"
        print(f"DEBUG: Boosted search term: {search_term}")

    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3", device=DEVICE
    )
    collection = client.get_collection(
        name="f1_2026_regulations", embedding_function=embedding_func
    )

    n_recall = 15 if state["loop_count"] == 0 else 25
    results = collection.query(query_texts=[search_term], n_results=n_recall)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)
    pairs = [[state["query"], chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    scored_results = sorted(
        zip(scores, chunks, metadatas), key=lambda x: x[0], reverse=True
    )
    top_results = scored_results[:4]

    return {
        "context": [item[1] for item in top_results],
        "metadatas": [item[2] for item in top_results],
        "loop_count": state["loop_count"] + 1,
    }


def transform_query_node(state: AgentState) -> Command:
    """
    NODE: Transform Query
    Rethinks the search strategy based on the 'Grade' of previous results.
    """
    print("--- NODE: TRANSFORM QUERY ---")

    strategy_hint = ""
    if state["grade"] == "partial":
        strategy_hint = (
            "CRITICAL HINT: You have the technical limit but are missing the PENALTY. "
            "STOP searching for specific car parts. Search for GENERAL SANCTIONS, "
            "DISQUALIFICATION rules, or Section A/B Breach consequences."
        )

    prompt = (
        f"Original User Query: {state['query']}\n"
        f"Last Search Result Grade: {state['grade']}\n\n"
        f"{strategy_hint}\n\n"
        "Generate a high-precision search query for the 2026 FIA regulations. "
        "Output ONLY the query string, no explanation or quotes."
    )

    response = chat(model="gemma4:e4b", messages=[{"role": "user", "content": prompt}])
    new_query = response.message.content.strip().replace('"', "")

    print(f"DEBUG: Pivoting search strategy to: {new_query}")

    return Command(update={"transformed_query": new_query}, goto="retrieve_node")


def grader_node(state: AgentState) -> Command:
    """
    NODE: Grader
    Evaluates context. Fixed to handle 'Chatty LLM' responses and substring traps.
    """
    print("--- NODE: GRADER ---")

    context_str = "\n".join(state["context"])
    prompt = (
        f"Query: {state['query']}\n"
        f"Context: {context_str}\n\n"
        "Critically evaluate if the context allows for a complete answer:\n"
        "1. If it has BOTH the rule AND the specific penalties, reply 'RELEVANT'.\n"
        "2. If it has only one part, reply 'PARTIAL'.\n"
        "3. If it has neither, reply 'IRRELEVANT'.\n"
        "Reply ONLY with the word."
    )

    response = chat(model="gemma4:e4b", messages=[{"role": "user", "content": prompt}])
    grade_raw = response.message.content.strip().upper()

    print(f"DEBUG: Raw Grader Output: {grade_raw}")

    if "RELEVANT" in grade_raw and "IRRELEVANT" not in grade_raw:
        print("DEBUG: Logic Gate -> GENERATE")
        return Command(update={"grade": "relevant"}, goto="generate_node")

    elif state["loop_count"] < 3:
        determined_grade = (
            "partial"
            if "PARTIAL" in grade_raw or "MISSING" in grade_raw
            else "irrelevant"
        )
        print(f"DEBUG: Logic Gate -> TRANSFORM ({determined_grade})")
        return Command(update={"grade": determined_grade}, goto="transform_query_node")

    else:
        print("DEBUG: Logic Gate -> FORCE GENERATE (Max Loops)")
        return Command(update={"grade": "failed"}, goto="generate_node")


def generate_node(state: AgentState):
    """
    NODE: Generator
    Synthesizes the answer by connecting technical limits to legal sanctions.
    """
    print("--- NODE: GENERATOR ---")

    formatted_context = ""
    for doc, meta in zip(state["context"], state["metadatas"]):
        formatted_context += f"--- RULE {meta.get('rule_id')} ---\n{doc}\n\n"

    system_prompt = (
        "You are an expert F1 Technical Delegate. Connect technical limits with legal consequences.\n\n"
        "INSTRUCTIONS:\n"
        "1. Cross-reference technical limits (Section C) with general penalties (Section A/B).\n"
        "2. Cite specific Rule IDs for both the limit and the penalty.\n"
        "3. If the context is still missing the penalty, admit it clearly.\n\n"
        f"REGULATORY CONTEXT:\n{formatted_context}"
    )

    response = chat(
        model="gemma4:e4b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["query"]},
        ],
    )

    return {"answer": response.message.content}

import re
import torch
from ollama import chat
from langgraph.types import Command
from sentence_transformers import CrossEncoder
from .state import AgentState

from ..db_client import get_collection, fetch_full_rule


def get_device():
    """Detects hardware accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

_RERANKER = None


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        print("DEBUG: Initializing BGE-Reranker into MPS Memory...")
        _RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)
    return _RERANKER


def retrieve_node(state: AgentState):
    """NODE: Retriever - High-recall semantic search followed by precision reranking."""
    print(f"\n--- NODE: RETRIEVER (Loop: {state['loop_count']}) ---")

    search_term = state.get("transformed_query") or state["query"]
    collection = get_collection()

    n_recall = 15 if state["loop_count"] == 0 else 25
    results = collection.query(query_texts=[search_term], n_results=n_recall)

    reranker = get_reranker()
    pairs = [[state["query"], chunk] for chunk in results["documents"][0]]
    scores = reranker.predict(pairs)

    scored_results = sorted(
        zip(scores, results["documents"][0], results["metadatas"][0]),
        key=lambda x: x[0],
        reverse=True,
    )

    top_results = scored_results[:5]

    return {
        "context": [item[1] for item in top_results],
        "metadatas": [item[2] for item in top_results],
        "loop_count": state["loop_count"] + 1,
    }


def cross_reference_node(state: AgentState):
    """NODE: Cross-Referencer - Deterministic enrichment via explicit regulatory citations."""
    print("--- NODE: CROSS-REFERENCER ---")

    chunks_to_scan = state["context"][:3]
    existing_rule_ids = {
        str(m.get("rule_id")) for m in state["metadatas"] if m.get("rule_id")
    }

    new_context, new_metadatas = [], []

    pattern = r"Article\s+([A-F]?\d+(?:\.\d+)*)"

    for text in chunks_to_scan:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_match = match.strip().upper()

            if any(clean_match in eid for eid in existing_rule_ids):
                continue

            fetched_data = fetch_full_rule(clean_match)
            if fetched_data:
                print(
                    f"SUCCESS: Enriching context with referenced Article {clean_match}"
                )
                for doc, meta in fetched_data:
                    new_context.append(doc)
                    new_metadatas.append(meta)
                    existing_rule_ids.add(str(meta.get("rule_id")))
            else:
                existing_rule_ids.add(clean_match)

    return {"context": new_context, "metadatas": new_metadatas}


def grader_node(state: AgentState) -> Command:
    """NODE: Grader - Verifies evidentiary sufficiency."""
    print("--- NODE: GRADER ---")

    context_str = "\n".join(state["context"])
    prompt = (
        f"Query: {state['query']}\n"
        f"Context: {context_str}\n\n"
        "Evaluate if the provided context contains the necessary regulatory evidence to answer the query.\n"
        "1. RELEVANT: The context contains the specific articles, rules, or definitions requested.\n"
        "2. PARTIAL: The context is related but lacks specific referenced details.\n"
        "3. IRRELEVANT: The context does not address the query.\n"
        "Respond ONLY with one word: RELEVANT, PARTIAL, or IRRELEVANT."
    )

    response = chat(model="gemma4:e4b", messages=[{"role": "user", "content": prompt}])
    grade = response.message.content.strip().upper()
    print(f"DEBUG: Evidence Grade -> {grade}")

    if "RELEVANT" in grade and "IRRELEVANT" not in grade:
        return Command(goto="generate_node")
    elif state["loop_count"] < 3:
        return Command(update={"grade": grade.lower()}, goto="transform_query_node")

    return Command(goto="generate_node")


def transform_query_node(state: AgentState) -> Command:
    """NODE: Transform Query - Multi-perspective re-searching."""
    print("--- NODE: TRANSFORM QUERY ---")

    prompt = (
        f"Query: {state['query']}\n"
        f"Context Sufficiency: {state['grade']}\n\n"
        "Generate a high-precision search query for FIA regs. Output ONLY the query string."
    )

    response = chat(model="gemma4:e4b", messages=[{"role": "user", "content": prompt}])
    return Command(
        update={"transformed_query": response.message.content.strip()},
        goto="retrieve_node",
    )


def generate_node(state: AgentState):
    """NODE: Generator - Synthesis of regulatory evidence with breadcrumb footer."""
    print("--- NODE: GENERATOR ---")

    formatted_context = ""
    for doc, meta in zip(state["context"], state["metadatas"]):
        formatted_context += f"--- RULE {meta.get('rule_id')} ---\n{doc}\n\n"

    system_prompt = (
        "You are the FIA Regulatory Intelligence System. Answer with extreme brevity and precision.\n\n"
        "STRICT CONSTRAINTS:\n"
        "1. Provide ONLY the direct, technical answer. No introductions or disclaimers.\n"
        "2. Use ONLY the provided context. If data is missing, state 'Information not found in regulations.'\n"
        "3. Cite every Rule ID used (e.g., [C4.1]).\n"
        "4. Avoid indirect or speculative data.\n\n"
        f"CONTEXT:\n{formatted_context}"
    )

    response = chat(
        model="gemma4:e4b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["query"]},
        ],
    )

    footer = "\n\n---\n**Sources used:**"
    seen_breadcrumbs = set()
    for meta in state["metadatas"]:
        bc = meta.get("breadcrumb")
        src = meta.get("source_pdf")
        pg = meta.get("page_number")
        if bc and bc not in seen_breadcrumbs:
            footer += f"\n* {bc} (Source: {src}, Page: {pg})"
            seen_breadcrumbs.add(bc)

    return {"answer": response.message.content + footer}

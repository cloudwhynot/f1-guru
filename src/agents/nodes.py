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


def planner_node(state: AgentState):
    """NODE: Planner - Decomposes the audit into Section-specific tasks."""
    print("--- NODE: PLANNER ---")

    history_context = state.get("summary", "This is the start of the conversation.")

    prompt = (
        "You are the 2026 FIA F1 Regulations Expert. Decompose the user query into "
        "2 strategic search tasks using ONLY these 2026 FIA F1 Regulations:\n"
        "- SECTION A: GENERAL REGULATORY PROVISIONS\n"
        "- SECTION B: SPORTING REGULATIONS\n"
        "- SECTION C: TECHNICAL REGULATIONS\n"
        "- SECTION D: FINANCIAL REGULATIONS (F1 TEAMS)\n"
        "- SECTION E: FINANCIAL REGULATIONS (POWER UNIT MANUFACTURERS)\n"
        "- SECTION F: OPERATIONAL REGULATIONS\n\n"
        f"CONVERSATION SUMMARY: {history_context}\n"
        f"USER QUERY: {state['query']}\n\n"
        "TASK INSTRUCTIONS:\n"
        "1. Generate exactly 2 high-precision keyword search strings (e.g., 'Section C car mass minimum').\n"
        "2. STRICT: Use ONLY FIA F1 terminology.\n"
        "3. Output ONLY a bulleted list of 2 tasks. No introductory text."
    )

    response = chat(model="gemma4:e4b", messages=[{"role": "user", "content": prompt}])

    tasks = [
        line.strip("- *").strip()
        for line in response.message.content.split("\n")
        if line.strip()
    ]

    for i, task in enumerate(tasks, 1):
        print(f"    {i}. {task}")

    return {"plan": tasks, "tasks_done": 0}


def retrieve_node(state: AgentState):
    """NODE: Retriever - High-recall semantic search followed by precision reranking."""
    print(f"\n--- NODE: RETRIEVER (Loop: {state['loop_count'] + 1}) ---")

    current_task_idx = state.get("tasks_done", 0)
    plan = state.get("plan", [])

    if plan and current_task_idx < len(plan):
        search_term = plan[current_task_idx]
        print(
            f"\n--- NODE: RETRIEVER (Task {current_task_idx + 1}/{len(plan)}: {search_term}) ---"
        )
    else:
        search_term = state.get("transformed_query") or state["query"]
        print(f"\n--- NODE: RETRIEVER (Loop: {state['loop_count'] + 1}) ---")

    if state["loop_count"] > 0 and not plan:
        search_term = f"FIA sanctions penalties breach non-compliance technical regulation {search_term}"
        print(f"DEBUG: Boosted search term: {search_term}")

    collection = get_collection()

    n_recall = 10 if state["loop_count"] == 0 else 20
    results = collection.query(query_texts=[search_term], n_results=n_recall)

    reranker = get_reranker()
    pairs = [[search_term, chunk] for chunk in results["documents"][0]]
    scores = reranker.predict(pairs)

    scored_results = sorted(
        zip(scores, results["documents"][0], results["metadatas"][0]),
        key=lambda x: x[0],
        reverse=True,
    )

    top_results = scored_results[:3]

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

    current_task_idx = state.get("tasks_done", 0)
    plan = state.get("plan", [])
    target = (
        plan[current_task_idx]
        if plan and current_task_idx < len(plan)
        else state["query"]
    )

    context_str = "\n".join(state["context"])
    prompt = (
        f"Goal: {target}\n"
        f"Context: {context_str}\n\n"
        "Evaluate if the provided context contains the necessary regulatory evidence to answer the query.\n"
        "1. RELEVANT: The context contains the specific articles, rules, or definitions requested.\n"
        "2. PARTIAL: The context is related but lacks specific referenced details.\n"
        "3. IRRELEVANT: The context does not address the query.\n"
        "CRITICAL: You are a binary classifier. You are forbidden from using sentences. Output ONLY: 'RELEVANT', 'PARTIAL', or 'IRRELEVANT'."
    )

    response = chat(model="gemma4:e4b", messages=[{"role": "user", "content": prompt}])

    grade_upper = response.message.content.strip().upper()
    print(f"DEBUG: Evidence Grade -> {grade_upper}")

    is_relevant = "RELEVANT" in grade_upper and "IRRELEVANT" not in grade_upper

    if is_relevant:
        new_tasks_done = current_task_idx + 1

        if plan and new_tasks_done < len(plan):
            print(
                f"DEBUG: Task {new_tasks_done} complete. Progressing to Task {new_tasks_done + 1}."
            )
            return Command(
                update={"tasks_done": new_tasks_done, "loop_count": 0},
                goto="retrieve_node",
            )
        return Command(goto="generate_node")
    elif state["loop_count"] < 3:
        return Command(
            update={"grade": grade_upper.lower()}, goto="transform_query_node"
        )

    return Command(goto="generate_node")


def transform_query_node(state: AgentState) -> Command:
    """NODE: Transform Query - Multi-perspective re-searching."""
    print("--- NODE: TRANSFORM QUERY ---")

    current_task_idx = state.get("tasks_done", 0)
    plan = state.get("plan", [])
    target = (
        plan[current_task_idx]
        if plan and current_task_idx < len(plan)
        else state["query"]
    )

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


def summarizer_node(state: AgentState):
    """NODE: Summarizer - Compresses the latest turn into the rolling summary."""
    print("--- NODE: SUMMARIZER ---")

    clean_answer = state.get("answer", "")

    prompt = (
        f"Existing Summary: {state.get('summary', '')}\n"
        f"Latest User Query: {state['query']}\n"
        f"Latest Guru Answer: {clean_answer}\n\n"
        "Update the existing summary to include the key technical topics discussed. "
        "Keep it under 2 sentences. Focus on Rule IDs mentioned."
    )

    response = chat(model="gemma4:e4b", messages=[{"role": "user", "content": prompt}])
    return {"summary": response.message.content.strip()}


def generate_node(state: AgentState):
    """NODE: Generator - Synthesis of regulatory evidence with breadcrumb footer."""
    print("--- NODE: GENERATOR ---")

    formatted_context = ""
    for doc, meta in zip(state["context"], state["metadatas"]):
        rule_id = str(meta.get("rule_id", "N/A"))
        formatted_context += f"--- RULE {rule_id} (\n{doc}\n\n"

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
    seen = set()
    for meta in state["metadatas"]:
        bc = meta.get("breadcrumb")
        if bc and bc not in seen:
            footer += f"\n* {bc} (Source: {meta.get('source_pdf')}, Page: {meta.get('page_number')})"
            seen.add(bc)

    return {"answer": response.message.content, "sources_footer": footer}

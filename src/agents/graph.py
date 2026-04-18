from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import (
    planner_node,
    retrieve_node,
    grader_node,
    generate_node,
    transform_query_node,
    cross_reference_node,
    summarizer_node,
)


def build_f1_guru_graph():
    """
    Constructs the Agentic RAG Graph.
    Uses modern 2026 Command-based routing for dynamic loops.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("cross_reference_node", cross_reference_node)
    workflow.add_node("grader_node", grader_node)
    workflow.add_node("transform_query_node", transform_query_node)
    workflow.add_node("summarizer_node", summarizer_node)
    workflow.add_node("generate_node", generate_node)

    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "retrieve_node")
    workflow.add_edge("retrieve_node", "cross_reference_node")
    workflow.add_edge("cross_reference_node", "grader_node")
    workflow.add_edge("generate_node", "summarizer_node")
    workflow.add_edge("summarizer_node", END)

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


app = build_f1_guru_graph()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import retrieve_node, grader_node, generate_node, transform_query_node


def build_f1_guru_graph():
    """
    Constructs the Agentic RAG Graph.
    Uses modern 2026 Command-based routing for dynamic loops.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grader_node", grader_node)
    workflow.add_node("transform_query_node", transform_query_node)
    workflow.add_node("generate_node", generate_node)

    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "grader_node")

    workflow.add_edge("generate_node", END)

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


app = build_f1_guru_graph()

import gradio as gr
from src.agents.graph import app
from langgraph.types import Overwrite
import gradio as gr
from src.agents.graph import app
from langgraph.types import Overwrite


def format_sidebar_sources(metadatas):
    """
    Sanitizes and formats metadata for the sidebar.
    Prevents Markdown header injection.
    """
    if not metadatas:
        return "### 📚 Validated Sources\n*No technical evidence cited yet.*"

    unique_sources = []
    seen = set()

    for m in metadatas:
        bc = m.get("breadcrumb")
        src = m.get("source_pdf", "Unknown Source")
        pg = m.get("page_number", "N/A")

        if bc and bc not in seen:
            clean_bc = str(bc).lstrip("# ").strip()
            unique_sources.append(
                f"🔹 **{clean_bc}**\n&nbsp;&nbsp;&nbsp;*{src}* (Pg. {pg})"
            )
            seen.add(bc)

    if not unique_sources:
        return "### 📚 Validated Sources\n*No technical evidence cited yet.*"

    return "### 📚 Validated Sources\n\n" + "\n\n---\n\n".join(unique_sources)


def run_audit(message, history):
    prev_summary = ""
    if history:
        for msg in reversed(history):
            if msg["role"] == "assistant" and "metadata" in msg:
                prev_summary = msg["metadata"].get("summary", "")
                break

    initial_state = {
        "query": message,
        "plan": [],
        "tasks_done": 0,
        "transformed_query": "",
        "summary": prev_summary,
        "context": Overwrite([]),
        "metadatas": Overwrite([]),
        "answer": "",
        "sources_footer": "",
        "grade": "",
        "loop_count": 0,
    }

    config = {"configurable": {"thread_id": "f1_gradio_session"}}
    current_answer = "Processing audit..."

    for event in app.stream(initial_state, config=config):
        for state in event.items():
            if state is None:
                continue

            full_state = app.get_state(config).values

            if "answer" in state:
                current_answer = state["answer"]
            elif "answer" in full_state:
                current_answer = full_state["answer"]

            node_metas = full_state.get("metadatas", [])
            sources_display = format_sidebar_sources(node_metas)

            node_summary = full_state.get("summary", prev_summary)

            yield (
                {
                    "role": "assistant",
                    "content": current_answer,
                    "metadata": {
                        "summary": node_summary,
                    },
                },
                node_summary,
                sources_display,
            )


with gr.Blocks(fill_height=True, title="F1-Guru Auditor") as demo:
    with gr.Sidebar(label="Audit Intelligence", open=True):
        gr.Markdown("### 🏁 Rolling Summary")
        summary_out = gr.Markdown("Audit summary will appear here.")
        gr.Markdown("---")
        sources_out = gr.Markdown("Waiting for validated evidence...")

    gr.HTML("<h1 style='text-align: center;'>🏎️ F1-Guru: Technical Auditor</h1>")
    gr.ChatInterface(
        fn=run_audit,
        additional_outputs=[summary_out, sources_out],
        examples=[
            "What is the minimum weight of the car?",
        ],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

import sys
from pathlib import Path
from langgraph.types import Overwrite

sys.path.append(str(Path(__file__).parent))

from src.agents.graph import app


def run_guru():
    print("Welcome to F1-Guru: 2026 Technical Auditor")
    print("-" * 40)

    config = {"configurable": {"thread_id": "f1_delegate_01"}}

    current_summary = ""

    while True:
        user_input = input("\n[USER]: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        initial_state = {
            "query": user_input,
            "plan": [],
            "tasks_done": 0,
            "transformed_query": "",
            "summary": current_summary,
            "context": Overwrite([]),
            "metadatas": Overwrite([]),
            "answer": "",
            "sources_footer": "",
            "grade": "",
            "loop_count": 0,
        }

        for event in app.stream(initial_state, config=config):
            for node, state in event.items():
                print(f"--- Finished executing: {node} ---")

                if node == "cross_reference_node":
                    fetched_ids = set(
                        [
                            m.get("rule_id")
                            for m in state.get("metadatas", [])
                            if m.get("rule_id")
                        ]
                    )
                    print(f"    > Context now contains rules: {list(fetched_ids)}")

        final_state = app.get_state(config)
        current_summary = final_state.values.get("summary", "")
        full_output = f"{final_state.values['answer']}\n{final_state.values.get('sources_footer', '')}"
        print(f"\n[GURU]: {full_output}")


if __name__ == "__main__":
    run_guru()

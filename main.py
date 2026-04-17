import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.agents.graph import app


def run_guru():
    print("Welcome to F1-Guru: 2026 Technical Auditor")
    print("-" * 40)

    config = {"configurable": {"thread_id": "f1_delegate_01"}}

    while True:
        user_input = input("\n[USER]: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        initial_state = {
            "query": user_input,
            "context": [],
            "metadatas": [],
            "answer": "",
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
        print(f"\n[GURU]: {final_state.values['answer']}")


if __name__ == "__main__":
    run_guru()

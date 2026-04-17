from typing import List, TypedDict, Annotated


def unique_add(left: list, right: list) -> list:
    """Reducer: Combines lists while keeping only unique elements."""
    new_list = list(left)
    for item in right:
        if item not in new_list:
            new_list.append(item)
    return new_list


class AgentState(TypedDict):
    query: str
    transformed_query: str
    context: Annotated[List[str], unique_add]
    metadatas: Annotated[List[dict], unique_add]
    answer: str
    grade: str
    loop_count: int

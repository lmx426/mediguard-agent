"""Case Agent lightweight session summary."""


def next_summary(previous: str, question: str, answer: str) -> str:
    """Append the latest turn to the bounded working-memory summary."""

    item = f"问：{question[:120]}；答：{answer[:160]}"
    if not previous:
        return item
    merged = f"{previous}\n{item}"
    return merged[-1200:]

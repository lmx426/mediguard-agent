"""模型网关端口。

定义与 LLM 模型通信的抽象接口，与具体供应商实现解耦。
"""

from __future__ import annotations

from typing import Any, Protocol


class ModelGateway(Protocol):
    """OpenAI-compatible chat completion boundary."""

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        require_json: bool = False,
        model: str | None = None,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> Any: ...

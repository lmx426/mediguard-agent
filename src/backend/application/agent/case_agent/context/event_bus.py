"""Local event bus for Caser context refreshes."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CaserContextEvent:
    """A local event requesting one or more Caser sections to refresh."""

    event_type: str
    case_id: str
    section_keys: list[str] = field(default_factory=list)
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class CaserContextEventBus:
    """Small in-process event bus used before introducing a real queue."""

    def __init__(self, max_workers: int = 2) -> None:
        self._handlers: list[Callable[[CaserContextEvent], None]] = []
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="caser-context",
        )

    def subscribe(self, handler: Callable[[CaserContextEvent], None]) -> None:
        self._handlers.append(handler)

    def publish(
        self,
        event_type: str,
        case_id: str,
        *,
        section_keys: list[str] | None = None,
        reason: str = "",
        payload: dict[str, Any] | None = None,
        async_refresh: bool = True,
    ) -> None:
        event = CaserContextEvent(
            event_type=event_type,
            case_id=case_id,
            section_keys=list(section_keys or []),
            reason=reason,
            payload=payload or {},
        )
        for handler in list(self._handlers):
            if async_refresh:
                self._executor.submit(handler, event)
            else:
                handler(event)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

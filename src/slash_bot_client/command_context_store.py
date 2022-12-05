from typing import Optional

from interactions import CommandContext

__context_store = {}


def get(key: str) -> Optional[CommandContext]:
    return __context_store.get(key, None)


def put(key: Optional[str], value: CommandContext) -> None:
    if key is not None:
        __context_store[key] = value

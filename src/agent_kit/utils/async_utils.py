from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Coroutine
from typing import Any, TypeVar, cast, overload

T = TypeVar("T")


def is_awaitable(value: object) -> bool:
    return inspect.isawaitable(value)


def run_coroutine_sync(coro: Awaitable[T]) -> T:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(cast(Coroutine[Any, Any, T], coro))
    if not loop.is_running():
        return loop.run_until_complete(coro)

    result_box: dict[str, T] = {}
    error_box: dict[str, Exception] = {}

    def runner() -> None:
        try:
            result_box["value"] = asyncio.run(cast(Coroutine[Any, Any, T], coro))
        except Exception as exc:  # pragma: no cover - defensive
            error_box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_box:
        raise error_box["error"]
    return result_box["value"]


async def maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value


@overload
def maybe_await_sync(value: Awaitable[T]) -> T:
    ...


@overload
def maybe_await_sync(value: T) -> T:
    ...


def maybe_await_sync(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return run_coroutine_sync(cast(Awaitable[T], value))
    return value

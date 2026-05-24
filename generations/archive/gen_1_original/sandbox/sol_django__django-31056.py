import asyncio
import contextvars
from typing import Any, AsyncIterator, Iterable, List, TypeVar, Union

from asgiref.local import Local
from django.core.exceptions import SynchronousOnlyOperation
from django.db.models import QuerySet
from unittest import mock

_state = contextvars.ContextVar("state", default=None)


def is_async_context() -> bool:
    """
    Determine if the current context is asynchronous.

    This function checks if the current execution is happening inside
    an asynchronous event loop or if it's explicitly marked as async.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop is not None and loop.is_running()
    except RuntimeError:  # No running event loop
        pass
    # Check the contextvar in case running outside the loop
    # (or event loop is not set, e.g. in a pytest event loop)
    state = _state.get()
    if state is None:
        return False
    return state.get("is_async", False)


def mark_as_async():
    """Mark code as running in a async context."""
    state = _state.get() or {}
    state["is_async"] = True
    _state.set(state)


def mark_as_sync():
    """Mark code as running in a sync context."""
    state = _state.get() or {}
    state["is_async"] = False
    _state.set(state)


_T = TypeVar("_T")


async def async_qs_iterator(qs: QuerySet[_T]) -> AsyncIterator[_T]:
    """
    Async iterator for a QuerySet.
    """
    if not is_async_context():
        raise SynchronousOnlyOperation(
            "You cannot use async QuerySet evaluation in a synchronous context."
        )

    for item in qs:
        yield item


async def async_qs_to_list(qs: QuerySet[_T]) -> List[_T]:
    """
    Convert a QuerySet to an async list.
    """
    items = []
    async for item in async_qs_iterator(qs):
        items.append(item)
    return items


async def main():
    # Mock QuerySet for testing purposes.
    class MockModel:
        def __init__(self, id: int, name: str):
            self.id = id
            self.name = name

        def __repr__(self):
            return f"MockModel(id={self.id}, name='{self.name}')"

    class MockQuerySet(QuerySet):
        def __init__(self, iterable: Iterable[Any]):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

    # Test Case 1: Async iteration within async context

    async def test_async_iteration():
        mark_as_async()  # Explicitly mark as async context
        mock_data = [MockModel(1, "Item 1"), MockModel(2, "Item 2")]
        mock_qs = MockQuerySet(mock_data)
        result = []
        async for item in async_qs_iterator(mock_qs):
            result.append(item)
        assert result == mock_data, f"Expected {mock_data}, but got {result}"
        print("Async iteration test passed.")
        mark_as_sync()

    await test_async_iteration()

    # Test Case 2: Synchronous iteration should raise an error
    class MockQuerySetSync(QuerySet):
        def __init__(self, iterable: Iterable[Any]):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

    mock_data_sync = [MockModel(3, "Item 3"), MockModel(4, "Item 4")]
    mock_qs_sync = MockQuerySetSync(mock_data_sync)

    def test_sync_iteration_error():
        try:
            asyncio.run(async_qs_iterator(mock_qs_sync).__anext__())
        except SynchronousOnlyOperation:
            print("Synchronous iteration test (error) passed.")
        else:
            raise AssertionError("Expected SynchronousOnlyOperation error")

    test_sync_iteration_error()

if __name__ == "__main__":
    asyncio.run(main())
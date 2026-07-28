"""Test-process cleanup for the CPU-heavy second-order contract suites."""

import gc
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def release_test_graphs() -> Iterator[None]:
    yield
    gc.collect()

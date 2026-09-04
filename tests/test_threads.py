"""Holding the pipeline to one thread, and putting the counts back afterwards."""

from __future__ import annotations

import os

import pytest

from mrmotion import single_threaded, thread_counts


def test_every_library_is_held_to_one() -> None:
    """Asserted, not assumed: the counts are read back inside the block."""
    with single_threaded():
        inside = thread_counts()
    for library, count in inside.items():
        assert count in (1, None), f"{library} was at {count}"


def test_counts_are_restored() -> None:
    before = thread_counts()
    with single_threaded():
        pass
    assert thread_counts() == before


def test_counts_are_restored_after_a_failure() -> None:
    before = thread_counts()
    with pytest.raises(RuntimeError), single_threaded():
        raise RuntimeError("the block failed")
    assert thread_counts() == before


def test_the_environment_is_set_for_libraries_loaded_inside() -> None:
    with single_threaded():
        assert os.environ["OMP_NUM_THREADS"] == "1"


def test_the_environment_is_restored() -> None:
    before = os.environ.get("OMP_NUM_THREADS")
    with single_threaded():
        pass
    assert os.environ.get("OMP_NUM_THREADS") == before


def test_a_larger_count_is_allowed() -> None:
    with single_threaded(2):
        assert thread_counts()["torch"] in (2, None)


def test_zero_threads_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        with single_threaded(0):
            pass

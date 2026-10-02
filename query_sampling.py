"""Shared deterministic query-point sampling for benchmark datasets."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Iterable
from typing import Any

import numpy as np


def normalize_query_limit(value: int | None) -> int | None:
    """Normalize ``None``/``-1`` to all queries and validate positive limits."""

    if value is None:
        return None
    try:
        limit = operator.index(value)
    except TypeError as error:
        raise ValueError("query_limit must be a positive integer, -1, or None") from error
    if limit == -1:
        return None
    if limit < 1:
        raise ValueError("query_limit must be a positive integer, -1, or None")
    return int(limit)


def _seed_words(seed: int, epoch: int, sample_identity: Any) -> list[int]:
    """Encode all sampling identities into stable platform-independent words."""

    payload = json.dumps(
        [int(seed), int(epoch), sample_identity],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=16).digest()
    return np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()


def sample_uniform_query_indices(
    rng: np.random.Generator,
    population_size: int,
    query_count: int,
) -> np.ndarray:
    """Sample uniform discrete indices with the Darcy legacy replacement rule.

    A request no larger than the population is sampled without replacement.
    If ``query_count`` exceeds ``population_size``, the complete draw is made
    with replacement so the requested query count is preserved.
    """

    try:
        population = operator.index(population_size)
        requested = operator.index(query_count)
    except TypeError as error:
        raise ValueError("population_size and query_count must be positive integers") from error
    if population < 1 or requested < 1:
        raise ValueError("population_size and query_count must be positive integers")
    indices = rng.choice(
        population,
        size=requested,
        replace=population < requested,
    )
    return np.sort(indices).astype(np.int64, copy=False)


def query_selection(
    query_count: int,
    query_limit: int | None,
    *,
    random_queries: bool,
    seed: int,
    epoch: int,
    sample_identity: Any,
) -> tuple[slice | np.ndarray, np.ndarray]:
    """Return a data selection and its sorted query-pool indices.

    Random subsets are deterministic functions of ``seed``, ``epoch`` and the
    stable sample identity.  Requests up to the population size are sampled
    without replacement; larger requests use one all-with-replacement draw and
    therefore may contain repeated indices.
    """

    count = operator.index(query_count)
    if count < 1:
        raise ValueError("query_count must be a positive integer")
    limit_value = normalize_query_limit(query_limit)
    requested = count if limit_value is None else limit_value
    if random_queries and requested != count:
        sequence = np.random.SeedSequence(_seed_words(seed, epoch, sample_identity))
        indices = sample_uniform_query_indices(
            np.random.default_rng(sequence), count, requested
        )
        return indices, indices
    # Fixed selections cannot synthesize extra observations.  The random
    # ``requested == count`` fast path is also equivalent to a sorted full
    # without-replacement draw and avoids allocating a large permutation.
    limit = min(requested, count)
    selection = slice(0, limit)
    return selection, np.arange(limit, dtype=np.int64)


def set_epoch_recursive(dataset: Any, epoch: int) -> None:
    """Propagate an epoch to a dataset and common PyTorch dataset wrappers."""

    epoch_value = int(epoch)
    visited: set[int] = set()

    def visit(current: Any) -> None:
        if current is None or id(current) in visited:
            return
        visited.add(id(current))
        setter = getattr(current, "set_epoch", None)
        if callable(setter):
            setter(epoch_value)
        children: list[Any] = []
        for attribute in ("dataset", "base"):
            child = getattr(current, attribute, None)
            if child is not None:
                children.append(child)
        nested = getattr(current, "datasets", None)
        if isinstance(nested, Iterable):
            children.extend(nested)
        for child in children:
            visit(child)

    visit(dataset)


__all__ = [
    "normalize_query_limit",
    "query_selection",
    "sample_uniform_query_indices",
    "set_epoch_recursive",
]

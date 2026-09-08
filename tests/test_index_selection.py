"""Which documents go into the index, and how many.

Sampling is not a model wrapper, so unlike the retrievers it is worth asserting
directly: the experiment ladder varies index size alone, and a sample that
quietly tracked the corpus size instead would make "more neighbours" and "more
training signal" inseparable.
"""

import pipeline_fixture as corpus
import pytest

from llms4subjects.config import IndexConfig
from llms4subjects.stages.indexes import select_documents


@pytest.fixture(scope="module")
def records():
    return corpus.records(corpus.fixture()["index"])


def test_no_size_indexes_the_whole_corpus(records):
    assert select_documents(records, IndexConfig()) == list(records)


@pytest.mark.parametrize("size", [1, 17, 70])
def test_size_is_independent_of_the_corpus_size(records, size):
    """A size is an absolute number, not a share of whatever it was handed."""
    half = records[: len(records) // 2]

    assert len(select_documents(records, IndexConfig(size=size))) == size
    assert len(select_documents(half, IndexConfig(size=size))) == size


def test_a_size_larger_than_the_corpus_takes_all_of_it(records):
    selected = select_documents(records, IndexConfig(size=len(records) * 2))

    assert sorted(r.id for r in selected) == sorted(r.id for r in records)


def test_the_sample_is_reproducible(records):
    config = IndexConfig(size=40)

    assert [r.id for r in select_documents(records, config)] == [
        r.id for r in select_documents(records, config)
    ]


def test_the_seed_selects_a_different_sample(records):
    a = select_documents(records, IndexConfig(size=40, seed=1))
    b = select_documents(records, IndexConfig(size=40, seed=2))

    assert [r.id for r in a] != [r.id for r in b]


def test_stratification_keeps_the_cell_proportions(records):
    size = 50
    selected = select_documents(records, IndexConfig(size=size, stratify=True))

    assert len(selected) == size
    for cell, count in _cells(records).items():
        share = count / len(records)
        assert abs(_cells(selected).get(cell, 0) / size - share) <= 1 / size


def test_stratification_keeps_every_cell_that_fits(records):
    """A cell dropped entirely would remove a language from the index."""
    selected = select_documents(records, IndexConfig(size=8, stratify=True))

    assert set(_cells(selected)) == set(_cells(records))


def _cells(records) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        counts[(record.type, record.lang)] = counts.get((record.type, record.lang), 0) + 1
    return counts

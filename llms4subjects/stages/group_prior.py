"""A 66-way classifier over GND classification groups, as a score adjustment.

This is the one place a dense output layer fits. The vocabulary has no hierarchy
to propagate through — the release contains no `skos:broader` triples at all —
but every subject carries exactly one of 66 classification groups, and those
groups have roughly 485 tib-core train documents each. That is the opposite of
the situation that made the rejected 14,607-way head predict zero (see
`baseline/`): a label with one positive example against 32,000 negatives cannot
be learned, and a group with hundreds can.

The head is linear and sits on top of the document tower's frozen vectors
rather than being a fine-tune of its own. Three reasons, in the order they
matter: the encoder's vectors are already computed and cached for the index, so
the prior costs a matrix multiply rather than a GPU run — and docs/spec.md
reserves the GPU host for three named training runs, of which this is not one;
one linear layer over a frozen representation is the smallest thing that can
answer "does the group signal help", which is the question the ticket asks; and
persisting it is two arrays rather than a checkpoint, so a prior found on disk
months later still loads.

It is trained one-against-the-rest rather than as a single softmax, because a
record's gold labels can name more than one group — 15.9% of tib-core train
records span three or more — and a softmax would have to be told which one is
"the" answer. The probabilities are then normalised into a distribution, which
leaves the ordering untouched and makes the weight in `apply_prior` mean the
same thing for a record concentrated on one group as for a record spread over
five.

The distribution **boosts** candidates in likely groups and never filters
unlikely ones (docs/spec.md, story 29): the 23.3% of multi-label records whose
labels span three or more groups would otherwise be permanently lost, and a
stage that can drop a candidate is a filter whatever it is called. So
`apply_prior` returns the same codes it was given, reordered.

Implemented by ticket 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import expit

from ..config import GroupPriorConfig
from ..contracts import Candidate, CandidateList, Code, Record, VocabularyEntry

# What `LogisticRegression` is given when a group has every document or none of
# them, in place of a fit it cannot perform. Large enough that the group is
# never the most likely one, finite so that it stays in the distribution: a
# group absent from a smaller index is unlikely, not unreachable.
DEGENERATE_LOGIT = 20.0

# lbfgs on 32,043 by 768 converges well inside this; it is a ceiling rather
# than a schedule.
MAX_ITERATIONS = 1000


def group_of_code(entries: Iterable[VocabularyEntry]) -> dict[Code, str]:
    """The classification group of every entry that names one.

    An entry with no classification name is left out rather than given a
    placeholder group: all 79,427 tib-core entries have one, so an absence
    means a vocabulary slice, and a boost reading a placeholder would be
    reading a guess.
    """
    return {
        entry.code: entry.classification_name
        for entry in entries
        if entry.classification_name
    }


def groups(entries: Iterable[VocabularyEntry]) -> tuple[str, ...]:
    """The classification groups the vocabulary names, in a fixed order.

    Sorted, because the order is the column order of the trained head: it is
    persisted beside the weights and has to mean the same thing on the next
    run, whatever order the vocabulary file happened to be in.
    """
    return tuple(sorted({entry.classification_name for entry in entries
                         if entry.classification_name}))


def record_groups(
    records: Sequence[Record], group_of: Mapping[Code, str]
) -> list[tuple[str, ...]]:
    """The groups a record's gold labels fall in — its training target.

    Gold labels, so this is for indexed training documents and for measuring
    the classifier's accuracy on a scored split. It never touches a record
    being predicted; `apply_prior` reads the classifier's output alone.
    """
    return [
        tuple(sorted({group_of[code] for code in record.subjects
                      if code in group_of}))
        for record in records
    ]


@dataclass(frozen=True)
class GroupPrior:
    """A trained one-against-the-rest head, as plain arrays.

    `coefficients` is one row per group in `groups` order, `intercepts` one
    scalar each. Storing the fit this way rather than as a fitted estimator is
    what lets the artifact outlive the scikit-learn version that produced it.
    """

    groups: tuple[str, ...]
    coefficients: np.ndarray
    intercepts: np.ndarray

    @property
    def dimensions(self) -> int:
        return int(self.coefficients.shape[1])

    def distributions(self, vectors: np.ndarray) -> list[dict[str, float]]:
        """A distribution over every group, per row of `vectors`.

        Every group appears in every row, including the ones no document
        trained them: a caller that read a missing key as "no boost" and a
        present zero as "no boost" would agree, but one that read a missing key
        as an error would not, and the boost has 66 keys to look up per
        candidate.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"the prior was trained on {self.dimensions} dimensions and "
                f"was given vectors of shape {vectors.shape}; a prior trained "
                "on one encoder's vectors cannot read another's"
            )

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            # Accelerate's BLAS raises the divide-by-zero and overflow flags on
            # matmuls whose inputs and results are entirely finite (numpy 2.1
            # on Apple Silicon), and this is the local host. The flags say
            # nothing about this product, so they are not read.
            logits = vectors @ self.coefficients.T + self.intercepts
        # `expit` rather than `1 / (1 + exp(-x))`: a strongly negative logit
        # overflows the exponential and reaches the same answer through a
        # warning.
        probabilities = expit(logits)
        totals = probabilities.sum(axis=1, keepdims=True)
        # A row the head finds nothing in is uniform rather than a division by
        # zero: no group is preferred, which is the honest reading of it.
        uniform = np.full_like(probabilities, 1.0 / len(self.groups))
        normalised = np.where(totals > 0.0, probabilities / np.where(
            totals > 0.0, totals, 1.0), uniform)

        return [
            {group: float(value) for group, value in zip(self.groups, row)}
            for row in normalised
        ]


def train(
    vectors: np.ndarray,
    group_sets: Sequence[Sequence[str]],
    all_groups: Sequence[str],
    config: GroupPriorConfig,
) -> GroupPrior:
    """Fit one logistic regression per group over frozen document vectors.

    `group_sets[i]` are the groups of document `i`'s gold labels, so a document
    is a positive for each of them. A document with no groups at all — every
    gold label outside the vocabulary being predicted over — is a negative
    everywhere rather than dropped, because "this text is none of the groups I
    can see" is a fact about the corpus the head should learn.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(group_sets) != len(vectors):
        raise ValueError(
            f"{len(vectors)} documents and {len(group_sets)} group sets; the "
            "targets are positional and would train the wrong column"
        )

    from sklearn.linear_model import LogisticRegression

    dimensions = vectors.shape[1] if len(vectors) else 0
    coefficients = np.zeros((len(all_groups), dimensions))
    intercepts = np.zeros(len(all_groups))

    memberships = [set(entry) for entry in group_sets]
    for column, group in enumerate(all_groups):
        target = np.array(
            [group in membership for membership in memberships], dtype=np.int8
        )
        positives = int(target.sum())
        if positives == 0 or positives == len(target):
            # One of the 66 groups has no tib-core train document, and a
            # smaller index has several. A constant answer keeps the column in
            # the distribution without pretending to a fit.
            intercepts[column] = DEGENERATE_LOGIT if positives else -DEGENERATE_LOGIT
            continue

        model = LogisticRegression(
            C=config.regularization, max_iter=MAX_ITERATIONS
        ).fit(vectors, target)
        coefficients[column] = model.coef_[0]
        intercepts[column] = float(model.intercept_[0])

    return GroupPrior(
        groups=tuple(all_groups),
        coefficients=coefficients,
        intercepts=intercepts,
    )


def apply_prior(
    candidates: Sequence[CandidateList],
    group_distributions: Sequence[Mapping[str, float]],
    group_of: Mapping[Code, str],
    config: GroupPriorConfig,
) -> list[CandidateList]:
    """Reorder each record's candidates towards the groups the prior likes.

    The adjustment is additive in units of the record's own score range:

        boosted = score + weight * P(group of the code) * (best - worst)

    which is the only form that means the same thing for the three score scales
    that can arrive here. A fused list carries reciprocal-rank sums around
    0.05, a lone dense retriever carries cosine similarities, a lone lexical one
    carries BM25; a fixed additive constant would be decisive for the first and
    invisible to the third, and a multiplicative factor would invert the
    ordering of any negative score. Scaling by the spread also gives `weight` a
    reading: at 1.0, a group the prior is certain of can move a candidate across
    the whole range of the list, and at 0.0 nothing moves at all.

    Codes are never added or dropped, so the candidate ceiling is fusion's and
    this stage cannot change it — only which of those candidates ranks where,
    which is what R@10 and the submission's 50 read.
    """
    if len(group_distributions) != len(candidates):
        raise ValueError(
            f"{len(candidates)} candidate lists and "
            f"{len(group_distributions)} group distributions; the prior is "
            "positional and would boost the wrong records"
        )

    return [
        _boosted(result, distribution, group_of, config.weight)
        for result, distribution in zip(candidates, group_distributions)
    ]


def _boosted(
    result: CandidateList,
    distribution: Mapping[str, float],
    group_of: Mapping[Code, str],
    weight: float,
) -> CandidateList:
    if not result.candidates or not weight:
        return result

    scores = [candidate.score for candidate in result.candidates]
    spread = max(scores) - min(scores)
    if not spread:
        # Nothing to move a candidate across. A ranking of equal scores is
        # ordered by the tie-break below and the prior would only relabel it.
        return result

    boosted = [
        Candidate(
            code=candidate.code,
            score=candidate.score
            + weight * spread * _probability(candidate.code, distribution, group_of),
            sources=candidate.sources,
        )
        for candidate in result.candidates
    ]
    # Ties broken by code, so no ranking depends on a dict — the same rule
    # fusion sorts by.
    boosted.sort(key=lambda candidate: (-candidate.score, candidate.code))
    return CandidateList(record_id=result.record_id, candidates=tuple(boosted))


def _probability(
    code: Code, distribution: Mapping[str, float], group_of: Mapping[Code, str]
) -> float:
    """How likely the prior finds this code's group. Unknown reads as unlikely."""
    group = group_of.get(code)
    if group is None:
        return 0.0
    return float(distribution.get(group, 0.0))


@dataclass(frozen=True)
class GroupAccuracy:
    """How often the prior's top groups contain a record's true groups.

    `any_within[n]` is the share of records with at least one true group inside
    the top n predicted, and `all_within[n]` the share with every true group
    inside it. Both are reported because they answer different questions: the
    first is whether the boost points at anything right at all, the second
    whether it points at everything right — and 76.7% of multi-label records
    confine all their labels to at most two groups, so `all_within[2]` is the
    figure the boost's headroom is bounded by.
    """

    records: int
    any_within: dict[int, float]
    all_within: dict[int, float]
    groups_per_record: float


def accuracy(
    group_distributions: Sequence[Mapping[str, float]],
    truth: Sequence[Sequence[str]],
    depths: Sequence[int] = (1, 2, 3),
) -> GroupAccuracy:
    """Score the classifier against the true groups of each record's gold labels.

    Records with no true group are skipped rather than counted as misses: their
    gold labels are outside the vocabulary being grouped, so there is nothing
    for the prior to have got right or wrong.
    """
    if len(group_distributions) != len(truth):
        raise ValueError(
            f"{len(group_distributions)} distributions and {len(truth)} truth "
            "sets; the comparison is positional"
        )

    scored = [
        (_ranked(distribution), set(actual))
        for distribution, actual in zip(group_distributions, truth)
        if actual
    ]
    if not scored:
        return GroupAccuracy(
            records=0,
            any_within={depth: 0.0 for depth in depths},
            all_within={depth: 0.0 for depth in depths},
            groups_per_record=0.0,
        )

    any_within: dict[int, float] = {}
    all_within: dict[int, float] = {}
    for depth in depths:
        hits = 0
        complete = 0
        for ranked, actual in scored:
            top = set(ranked[:depth])
            hits += bool(top & actual)
            complete += actual <= top
        any_within[depth] = hits / len(scored)
        all_within[depth] = complete / len(scored)

    return GroupAccuracy(
        records=len(scored),
        any_within=any_within,
        all_within=all_within,
        groups_per_record=sum(len(actual) for _, actual in scored) / len(scored),
    )


def _ranked(distribution: Mapping[str, float]) -> list[str]:
    """Groups most likely first, ties broken by name so the order is stable."""
    return [
        group
        for group, _ in sorted(
            distribution.items(), key=lambda item: (-item[1], item[0])
        )
    ]

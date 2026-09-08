"""The coverage curve: abstention as an analysis, never as an output.

Ticket 16 asks one question the official metric cannot ask. The submission
format is exactly 50 ranked codes, so a system has no way to say "I don't
know" — but the workflow this project exists to support is suggest-and-confirm,
where a system that declines the records it is worst at is more useful than one
that guesses everywhere. The curve answers it from scores already computed: as
the confidence threshold rises, how many records still get an answer, and how
much better are the answers?

What is asserted here is what would make that figure a lie:

- a row that reports a coverage it did not actually take, because a tie at the
  threshold was cut arbitrarily;
- a precision figure quoted without the ceiling the gold set sizes impose on it;
- recall on the answered records read as recall on the split, when abstention
  gives up the gold assignments of every record it declines;
- an "abstention" that reached into the ranking and shortened it, which would
  change the output contract rather than analyse it;
- a figure that could be lifted into a leaderboard comparison, which is why
  every rendering carries the marker and a test holds it there.

Nothing here loads a model. The curve is arithmetic over rankings and gold, and
the metric under it is the evaluator, which is tested against the organizers'
scorer elsewhere.
"""

import importlib.util
from pathlib import Path

import pytest

from llms4subjects.contracts import Candidate, CandidateList, Record
from llms4subjects.stages.evaluator import evaluate


def _script(name: str):
    """`scripts/` is not importable as a package, so it is loaded by path.

    Registered in `sys.modules` before it is executed, as in
    `tests/test_screen_encoders.py`: `@dataclass` resolves a class's
    annotations through the module it was defined in.
    """
    import sys

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


coverage = _script("coverage_curve")


FILLER = tuple(f"gnd:filler-{index}" for index in range(60))


def record(record_id: str, gold: tuple[str, ...], language: str = "en") -> Record:
    return Record(
        id=record_id,
        type="Article",
        lang=language,
        title=f"title {record_id}",
        abstract="",
        subjects=gold,
    )


def ranking(record_id: str, codes: tuple[str, ...], score: float) -> CandidateList:
    """One record's ranking, 50 long, every candidate at the same score.

    A flat score is what makes `reranker.confidence` — the mean of the top five —
    exactly `score`, so a test can say what a record's confidence is rather than
    computing it a second way and asserting the two agree.
    """
    filled = codes + tuple(code for code in FILLER if code not in codes)
    return CandidateList(
        record_id=record_id,
        candidates=tuple(
            Candidate(code=code, score=score, sources={"dense": rank})
            for rank, code in enumerate(filled[:50])
        ),
    )


@pytest.fixture
def split():
    """Ten records whose confidence tracks correctness exactly.

    The five confident ones hold their gold label first; the five unconfident
    ones do not hold it at all. That is the shape the curve exists to expose,
    and it makes every row's expected value arithmetic rather than a fixture
    constant.
    """
    records = [record(f"r{index}", (f"gnd:gold-{index}",)) for index in range(10)]
    rankings = [
        ranking(f"r{index}", (f"gnd:gold-{index}",), score=0.9 - index * 0.01)
        if index < 5
        else ranking(f"r{index}", (), score=0.5 - index * 0.01)
        for index in range(10)
    ]
    return records, rankings


def points_by_requested(points):
    return {point.requested: point for point in points}


# --- Full coverage, and the metric under the curve ---------------------------


def test_full_coverage_is_the_whole_split(split):
    records, rankings = split
    points = coverage.curve(records, rankings, bands={})

    full = points_by_requested(points)[1.0]
    assert full.coverage == 1.0
    assert full.records == len(records)


def test_the_curve_reports_a_row_at_full_coverage_first(split):
    """The row a reader compares every other row against, and the ticket asks
    for it explicitly: without it the figure has no baseline."""
    records, rankings = split
    points = coverage.curve(records, rankings, bands={})

    assert points[0].requested == 1.0
    assert 1.0 in points_by_requested(points)


def test_the_precision_at_full_coverage_is_the_evaluator_s(split):
    """The curve slices the record set; it does not compute its own metric."""
    records, rankings = split
    points = coverage.curve(records, rankings, bands={})

    report = evaluate(
        gold={record.id: record.subjects for record in records},
        predictions={result.record_id: result.codes for result in rankings},
        bands={},
        cells={record.id: (record.type, record.lang) for record in records},
        ks=(coverage.PRECISION_K, coverage.RECALL_K),
    )
    full = points_by_requested(points)[1.0]
    assert full.precision == pytest.approx(report.micro.precision(coverage.PRECISION_K))
    assert full.recall == pytest.approx(report.micro.recall(coverage.RECALL_K))


# --- What the curve claims ---------------------------------------------------


def test_coverage_falls_and_the_threshold_rises_together(split):
    records, rankings = split
    points = coverage.curve(records, rankings, bands={})

    coverages = [point.coverage for point in points]
    thresholds = [point.threshold for point in points]
    assert coverages == sorted(coverages, reverse=True)
    assert thresholds == sorted(thresholds)


def test_precision_rises_as_the_least_confident_records_are_declined(split):
    """The claim the figure makes, on data where it is true by construction."""
    records, rankings = split
    points = points_by_requested(coverage.curve(records, rankings, bands={}))

    assert points[0.5].precision > points[1.0].precision
    assert points[0.5].no_hit < points[1.0].no_hit


def test_a_flat_confidence_buys_nothing(split):
    """The control: when confidence says nothing, declining records cannot help.

    Without this, a curve that sorted by anything at all — including per-record
    precision itself — would look like a working confidence measure.
    """
    records, _ = split
    flat = [ranking(record.id, record.subjects, score=0.5) for record in records[:5]]
    flat += [ranking(record.id, (), score=0.5) for record in records[5:]]

    points = points_by_requested(coverage.curve(records, flat, bands={}))
    assert points[0.5].precision == pytest.approx(points[1.0].precision)
    assert points[0.5].coverage == 1.0


# --- The honesty of a row ----------------------------------------------------


def test_a_tie_at_the_threshold_is_kept_whole(split):
    """Records that are equally confident are answered or declined together.

    Cutting a tie to hit a requested coverage exactly would make the threshold
    a fiction: two records the system cannot tell apart would be reported as
    one answered and one declined.
    """
    records, _ = split
    tied = [ranking(record.id, record.subjects, score=0.5) for record in records[:8]]
    tied += [ranking(record.id, (), score=0.1) for record in records[8:]]

    points = points_by_requested(coverage.curve(records, tied, bands={}))
    assert points[0.5].records == 8
    assert points[0.5].coverage == pytest.approx(0.8)
    assert points[0.5].requested == 0.5


def test_the_precision_ceiling_travels_with_the_precision(split):
    """At one gold label a record, P@5 cannot exceed 0.2 however good the system.

    Precision figures in this project are always printed with the achievable
    maximum beside them, and on a curve the maximum moves with the retained
    subset — so a row that dropped it would invite exactly the misreading the
    rest of the results document is careful to prevent.
    """
    records, rankings = split
    points = coverage.curve(records, rankings, bands={})

    for point in points:
        assert point.achievable == pytest.approx(1 / coverage.PRECISION_K)
        assert point.precision <= point.achievable + 1e-9


def test_the_gold_assignments_given_up_are_reported(split):
    """Recall on the answered records is not recall on the split.

    A curve that reported only the first would show abstention as free, when
    what it costs is exactly the gold assignments of every declined record.
    """
    records, rankings = split
    points = points_by_requested(coverage.curve(records, rankings, bands={}))

    assert points[1.0].assignment_share == pytest.approx(1.0)
    assert points[0.5].assignment_share == pytest.approx(0.5)
    assert points[0.5].assignments == 5


# --- The output contract, which this analysis may not touch ------------------


def test_the_analysis_returns_measurements_and_never_a_ranking(split):
    """Abstention is an analysis over confidence, never a change to the output.

    docs/spec.md: the submission format has no representation for fewer than 50
    codes, so a curve that "applied" a threshold — truncating the declined
    records' rankings, dropping them from the sequence it was handed, or handing
    back a filtered one for a caller to submit — would be measuring a system
    this project does not ship. What comes back is `Point`s and nothing else,
    and the sequence handed in is the same sequence afterwards.
    """
    records, rankings = split
    before = list(rankings)

    points = coverage.curve(records, rankings, bands={})

    assert all(isinstance(point, coverage.Point) for point in points)
    assert [id(result) for result in rankings] == [id(result) for result in before]
    assert all(len(result.candidates) == 50 for result in rankings)


def test_a_record_with_no_gold_is_dropped_from_every_column(split):
    """The evaluator drops it, so the coverage column may not keep it.

    A record with no gold labels has no recall denominator. Counting it as
    answered while no precision figure can see it would put the two halves of a
    row in disagreement — and at full coverage the no-hit share would improve
    for a record nothing was ever right or wrong about.
    """
    records, rankings = split
    records = records + [record("r-nogold", ())]
    rankings = rankings + [ranking("r-nogold", (), score=0.99)]

    points = points_by_requested(coverage.curve(records, rankings, bands={}))
    assert points[1.0].records == 10
    assert points[1.0].coverage == 1.0
    assert points[1.0].no_hit == pytest.approx(0.5)


def test_a_repeated_record_id_is_refused(split):
    """`all_train` holds 45 ids filed twice; keying by id would score one
    filing's ranking against the other's gold and never say so."""
    records, rankings = split
    with pytest.raises(ValueError, match="more than once"):
        coverage.curve(records + [records[0]], rankings, bands={})


def test_no_records_is_an_empty_curve():
    assert coverage.curve([], [], bands={}) == ()


def test_a_record_without_a_ranking_is_refused(split):
    """Positional guessing is how one record's scores land on another's gold."""
    records, rankings = split
    with pytest.raises(ValueError):
        coverage.curve(records, rankings[:-1], bands={})


# --- The marker, which is an acceptance criterion ----------------------------


def test_every_rendering_says_it_is_outside_the_official_metric(split):
    records, rankings = split
    points = coverage.curve(records, rankings, bands={})

    text = coverage.render(points, "the fused ranking")
    assert coverage.OUTSIDE in text
    assert coverage.OUTSIDE in coverage.figure_title("the fused ranking")


def test_the_table_holds_one_row_per_requested_coverage(split):
    records, rankings = split
    points = coverage.curve(records, rankings, bands={})

    rows = [line for line in coverage.render(points, "x").splitlines() if line.startswith("| ")]
    # One header row, one separator, then the points.
    assert len(rows) == len(points) + 1


def test_the_figure_is_written_where_it_was_asked_for(split, tmp_path):
    records, rankings = split
    points = coverage.curve(records, rankings, bands={})

    path = tmp_path / "curve.png"
    coverage.figure(points, path, "the fused ranking")
    assert path.exists() and path.stat().st_size > 0


# --- No additional inference -------------------------------------------------


def test_a_reranking_config_with_no_cached_pass_is_refused(tmp_path):
    """The ticket's constraint, enforced rather than promised.

    The curve is produced from scores already computed. A reranking config whose
    scoring pass is not on disk would otherwise quietly load a cross-encoder and
    spend hours, so the harness refuses and names the command that produces it.
    """
    from llms4subjects.artifacts import ArtifactStore
    from llms4subjects.config import load_experiment

    config = load_experiment(
        Path(__file__).resolve().parent.parent / "configs" / "rung1-rerank-base.yaml"
    )
    store = ArtifactStore(tmp_path, data_revision="deadbeef")

    with pytest.raises(coverage.NoCachedPass) as error:
        coverage.cached_scores(config, store, "core_dev", 5354)
    assert "rerank_report.py" in str(error.value)


def test_the_refusal_names_the_command_that_would_fill_the_cache(tmp_path):
    """Including the sample: the cache is keyed by the record set too, so a
    refusal that named the full-split command would send a reader to spend
    hours on a pass this run still could not read."""
    from llms4subjects.artifacts import ArtifactStore
    from llms4subjects.config import load_experiment

    config = load_experiment(
        Path(__file__).resolve().parent.parent / "configs" / "rung1-rerank-base.yaml"
    )
    store = ArtifactStore(tmp_path, data_revision="deadbeef")
    command = "python scripts/rerank_report.py configs/rung1-rerank-base.yaml --sample 300"

    with pytest.raises(coverage.NoCachedPass) as error:
        coverage.cached_scores(config, store, "core_dev-sample", 300, command)
    assert command in str(error.value)


def test_the_gold_test_split_is_refused():
    assert coverage.main(["configs/rung1.yaml", "--split", "core_test"]) == 1

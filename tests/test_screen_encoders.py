"""The encoder screen: what makes four numbers a comparison.

Ticket 09 screens four off-the-shelf encoders at an 8,000-document index and
ranks them on dev micro Recall@10. The screen's job is not to compute a metric —
the evaluator does that, and it is tested against the organizers' scorer — but
to hold the comparison honest: one variable, four models, everything else equal,
and the wall clock recorded beside each score so a later cost claim has a
number under it.

So what is asserted here is the part that would silently ruin the result: a
config that differs from the others in more than its encoder, a ranking that
does not order what it says it orders. The encoders themselves are model
weights, and nothing here loads any.
"""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from llms4subjects.config import (
    EncoderConfig,
    ExperimentConfig,
    FusionConfig,
    IndexConfig,
    LabelTextConfig,
)
from llms4subjects.stages.evaluator import EvaluationReport, Metrics


def _script(name: str):
    """`scripts/` is not importable as a package, so it is loaded by path.

    Registered in `sys.modules` before it is executed: `@dataclass` resolves a
    class's annotations through the module it was defined in, and a module that
    is not there yet has none.
    """
    import sys

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


screen = _script("screen_encoders")


def experiment(name: str, model: str, **overrides) -> ExperimentConfig:
    config = ExperimentConfig(
        name=name,
        encoder=EncoderConfig(name=model),
        index=IndexConfig(size=8000, stratify=True),
        label_text=LabelTextConfig(),
        fusion=FusionConfig(),
    )
    return dataclasses.replace(config, **overrides)


def report(recall: float, bands: dict[str, float] | None = None) -> EvaluationReport:
    """A report with one k, which is all the ranking and the tables read."""
    metrics = Metrics(at_k={10: {"precision": 0.0, "recall": recall, "f1": 0.0}})
    return EvaluationReport(
        micro=metrics,
        official_macro=metrics,
        by_band={
            band: Metrics(at_k={10: {"precision": 0.0, "recall": value, "f1": 0.0}})
            for band, value in (bands or {}).items()
        },
        by_language={},
        by_type={},
        ks=(10,),
    )


def screened(name: str, recall: float, seconds: float = 1.0, **kwargs):
    return screen.Screened(
        config=experiment(name, f"lab/{name}"),
        dimensions=768,
        parameters=None,
        fused=report(recall),
        per_retriever={},
        load_seconds=0.5,
        retrieve_seconds=seconds,
        wrote=3,
        **kwargs,
    )


# --- One variable, or it is not a screen -------------------------------------


def test_configs_differing_only_in_their_encoder_are_comparable():
    configs = [experiment("a", "lab/one"), experiment("b", "lab/two")]

    assert screen.differences(configs) == {}


def test_a_different_index_makes_the_run_two_experiments():
    """The whole point of rung 1 is that index size is held fixed at 8,000."""
    configs = [
        experiment("a", "lab/one"),
        experiment("b", "lab/two", index=IndexConfig(size=32043)),
    ]

    assert "index" in screen.differences(configs)


def test_a_different_label_rendering_is_refused_too():
    configs = [
        experiment("a", "lab/one"),
        experiment("b", "lab/two", label_text=LabelTextConfig(bilingual=False)),
    ]

    assert "label_text" in screen.differences(configs)


def test_a_different_fusion_weight_is_refused():
    """A weight tuned for one encoder and not another would rank the tuning."""
    configs = [
        experiment("a", "lab/one"),
        experiment("b", "lab/two", fusion=FusionConfig(rrf_k=10)),
    ]

    assert "fusion" in screen.differences(configs)


def test_the_encoder_section_is_the_variable_and_never_a_difference():
    configs = [
        experiment("a", "lab/one"),
        experiment("b", "lab/two", encoder=EncoderConfig(name="lab/two", batch_size=8)),
    ]

    assert "encoder" not in screen.differences(configs)


def test_names_and_notes_may_differ():
    """Every config needs its own name; notes are prose for a reader."""
    configs = [
        experiment("a", "lab/one", notes="the baseline"),
        experiment("b", "lab/two", notes="twice the size"),
    ]

    assert screen.differences(configs) == {}


def test_two_configs_naming_the_same_encoder_are_refused():
    """Otherwise a row is silently a duplicate of another, and the screen lies."""
    configs = [experiment("a", "lab/one"), experiment("b", "lab/one")]

    with pytest.raises(screen.NotAScreen, match="lab/one"):
        screen.check(configs)


def test_one_config_is_not_a_screen():
    with pytest.raises(screen.NotAScreen, match="at least two"):
        screen.check([experiment("a", "lab/one")])


def test_check_names_every_section_that_differs():
    configs = [
        experiment("a", "lab/one"),
        experiment("b", "lab/two", index=IndexConfig(size=10), fusion=FusionConfig(rrf_k=1)),
    ]

    with pytest.raises(screen.NotAScreen) as error:
        screen.check(configs)

    assert "index" in str(error.value)
    assert "fusion" in str(error.value)


# --- The ranking -------------------------------------------------------------


def test_the_ranking_is_by_the_selection_metric_best_first():
    rows = [screened("a", 0.31), screened("b", 0.42), screened("c", 0.07)]

    assert [row.name for row in screen.rank(rows, 10)] == ["lab/b", "lab/a", "lab/c"]


def test_the_ranking_breaks_ties_by_name_rather_than_by_arrival():
    """A tie decided by argument order would move with the command line."""
    rows = [screened("b", 0.4), screened("a", 0.4)]

    assert [row.name for row in screen.rank(rows, 10)] == ["lab/a", "lab/b"]


def test_the_tables_hold_one_row_per_encoder():
    rows = screen.rank([screened("a", 0.31), screened("b", 0.42)], 10)

    table = screen.render_ranking(rows, 10)

    body = [line for line in table.splitlines() if line.startswith("| lab/")]
    assert [line.split(" | ")[0] for line in body] == ["| lab/b", "| lab/a"]


def test_the_cost_table_says_whether_a_timing_was_cold():
    """A warm cache reads back vectors, so its seconds are not a cost."""
    rows = [screened("a", 0.31, seconds=200.0), screened("b", 0.42, seconds=7.0)]
    rows[1] = dataclasses.replace(rows[1], wrote=0)

    table = screen.render_cost(rows)

    assert "computed 3" in table
    assert "warm" in table


def test_a_pass_that_computed_some_of_its_vectors_says_how_many():
    """Three matrices per pass, so finding one and computing two is not warm."""
    row = dataclasses.replace(screened("a", 0.31), wrote=2)

    assert row.cache == "computed 2"


# --- What the index is made of -----------------------------------------------


def record(record_id: str, kind: str, lang: str):
    from llms4subjects.contracts import Record

    return Record(id=record_id, type=kind, lang=lang, title="t", abstract="a")


def test_the_index_report_names_every_cell_in_the_corpus():
    """Stratification is the ticket's criterion, so the screen states it."""
    records = [
        record("1", "Book", "de"),
        record("2", "Book", "en"),
        record("3", "Article", "en"),
        record("4", "Book", "de"),
    ]
    config = experiment("a", "lab/one", index=IndexConfig(size=2, stratify=True))

    table = screen.render_stratification(records, config)

    assert "2 of 4 documents" in table
    for kind, lang in (("Book", "de"), ("Book", "en"), ("Article", "en")):
        assert f"| {kind} | {lang} |" in table


def test_a_different_input_length_is_refused_even_though_it_is_the_encoder():
    """How much of a document a model reads is the experiment's, not the model's."""
    configs = [
        experiment("a", "lab/one"),
        experiment("b", "lab/two", encoder=EncoderConfig(name="lab/two", max_length=256)),
    ]

    assert "encoder.max_length" in screen.differences(configs)


def test_a_different_batch_size_is_allowed():
    """It moves wall clock and nothing else, and a large model may need one."""
    configs = [
        experiment("a", "lab/one"),
        experiment("b", "lab/two", encoder=EncoderConfig(name="lab/two", batch_size=8)),
    ]

    assert screen.differences(configs) == {}

"""Contrastive fine-tuning, at the seams that do not need a GPU.

What is asserted here is everything that decides *what the model is shown*:
which codes become hard negatives, how pairs become batches, and which scores
the loss is allowed to count. The training loop itself is a torch loop over
those three, and it runs on the one machine this test suite cannot reach.
"""

import json

import pytest

from llms4subjects.contracts import Candidate, CandidateList, Record
from llms4subjects.finetune import (
    ADAPTER_MANIFEST,
    AdapterMismatch,
    TrainingExample,
    adapter_manifest,
    batches,
    examples,
    false_negative_mask,
    hard_negatives,
    lora_targets,
    read_adapter,
)


def record(id, subjects, title="a document"):
    return Record(id=id, type="Book", lang="de", title=title, abstract="",
                  subjects=tuple(subjects))


def candidates(record_id, codes):
    return CandidateList(
        record_id,
        tuple(
            Candidate(code=code, score=1.0 - position / 100)
            for position, code in enumerate(codes)
        ),
    )


# --- Mining ------------------------------------------------------------------


def test_hard_negatives_are_the_retriever_s_own_top_mistakes():
    mined = hard_negatives(["a", "b", "c", "d"], gold={"b"}, count=2)
    assert mined == ("a", "c")


def test_a_gold_code_is_never_a_negative_however_it_was_ranked():
    """The whole risk of mining: the confusions include the right answers."""
    assert hard_negatives(["a", "b"], gold={"a", "b"}, count=2) == ()


def test_mining_takes_the_hardest_it_has_when_it_has_too_few():
    assert hard_negatives(["a"], gold=set(), count=5) == ("a",)


# --- Pairs -------------------------------------------------------------------


def test_one_example_per_gold_assignment():
    built = examples(
        [record("r1", ["x", "y"])],
        {"r1": ["n1", "n2"]},
        negatives=1,
    )

    assert [(e.record_id, e.positive) for e in built] == [("r1", "x"), ("r1", "y")]
    assert built[0].negatives == ("n1",)


def test_a_record_with_no_mined_candidates_still_trains_on_in_batch_negatives():
    built = examples([record("r1", ["x"])], {}, negatives=4)

    assert built[0].negatives == ()


def test_an_unlabelled_record_contributes_nothing():
    assert examples([record("r1", [])], {"r1": ["n"]}, negatives=1) == []


def test_a_record_s_own_gold_is_dropped_from_its_mined_negatives():
    """Mining ran per record, but a positive of one pair is gold for both."""
    built = examples([record("r1", ["x", "y"])], {"r1": ["y", "n"]}, negatives=2)

    assert built[0].negatives == ("n",)


# --- Batches -----------------------------------------------------------------


def example(record_id, positive, negatives=()):
    return TrainingExample(
        record_id=record_id, text="t", positive=positive, negatives=tuple(negatives)
    )


def test_a_batch_never_holds_one_code_twice():
    """In-batch negatives call every other positive wrong, so a repeat lies."""
    built = list(
        batches(
            [example("r1", "x"), example("r2", "x"), example("r3", "y")],
            size=2,
            seed=0,
        )
    )

    for batch in built:
        assert len({item.positive for item in batch}) == len(batch)


def test_every_example_is_used_once_per_epoch():
    pairs = [example(f"r{n}", f"c{n}") for n in range(10)]
    built = list(batches(pairs, size=3, seed=0))

    assert sorted(item.positive for batch in built for item in batch) == sorted(
        item.positive for item in pairs
    )


def test_batching_is_seeded_so_a_run_is_repeatable():
    pairs = [example(f"r{n}", f"c{n}") for n in range(10)]
    first = [[i.positive for i in b] for b in batches(pairs, size=3, seed=7)]
    again = [[i.positive for i in b] for b in batches(pairs, size=3, seed=7)]

    assert first == again


def test_a_different_seed_gives_a_different_epoch():
    pairs = [example(f"r{n}", f"c{n}") for n in range(20)]
    first = [[i.positive for i in b] for b in batches(pairs, size=4, seed=1)]
    other = [[i.positive for i in b] for b in batches(pairs, size=4, seed=2)]

    assert first != other


# --- What the loss may count -------------------------------------------------


def test_a_column_that_is_gold_for_another_row_is_masked_out():
    """Otherwise the loss punishes the model for a correct assignment."""
    batch = [example("r1", "x"), example("r2", "y")]
    gold = {"r1": {"x", "y"}, "r2": {"y"}}

    masked = false_negative_mask(batch, ["x", "y"], gold)

    # Row 0's target is column 0; column 1 is also gold for r1, so it is masked.
    assert masked == [[False, True], [False, False]]


def test_a_row_s_own_target_is_never_masked():
    batch = [example("r1", "x")]
    masked = false_negative_mask(batch, ["x"], {"r1": {"x"}})

    assert masked == [[False]]


def test_a_shared_hard_negative_is_masked_for_whoever_it_is_gold_for():
    batch = [example("r1", "x", ["n"]), example("r2", "n")]
    gold = {"r1": {"x"}, "r2": {"n"}}

    masked = false_negative_mask(batch, ["x", "n"], gold)

    assert masked[1] == [False, False]  # row 1's own target
    assert masked[0] == [False, False]  # `n` is not gold for r1, so it counts


# --- LoRA targets ------------------------------------------------------------


def test_each_shortlisted_encoder_names_the_modules_lora_adapts():
    assert lora_targets("BAAI/bge-m3")
    assert lora_targets("Alibaba-NLP/gte-multilingual-base")


def test_an_unknown_architecture_is_refused_rather_than_guessed():
    """Guessing wrong trains nothing and reports a fine-tune that happened."""
    with pytest.raises(KeyError, match="some/other-encoder"):
        lora_targets("some/other-encoder")


# --- The adapter artifact ----------------------------------------------------


def test_an_adapter_records_the_weights_it_was_trained_over():
    written = adapter_manifest(
        base="BAAI/bge-m3",
        revision="abc123",
        data_revision="rev1",
        negatives="key1",
        hyperparameters={"epochs": 2},
        losses=[1.0, 0.5],
        seconds=12.5,
        host="nlp2",
    )

    assert written["base"] == "BAAI/bge-m3"
    assert written["revision"] == "abc123"
    assert written["losses"] == [1.0, 0.5]


def test_an_adapter_trained_over_another_encoder_is_refused(tmp_path):
    """The adapter is deltas onto specific weights; over others it is noise."""
    (tmp_path / ADAPTER_MANIFEST).write_text(
        json.dumps(
            adapter_manifest(
                base="BAAI/bge-m3",
                revision="abc123",
                data_revision="rev1",
                negatives="key1",
                hyperparameters={},
                losses=[],
                seconds=0.0,
                host="nlp2",
            )
        )
    )

    read_adapter(tmp_path, "BAAI/bge-m3", "abc123")
    with pytest.raises(AdapterMismatch, match="gte-multilingual-base"):
        read_adapter(tmp_path, "Alibaba-NLP/gte-multilingual-base", "abc123")
    with pytest.raises(AdapterMismatch, match="def456"):
        read_adapter(tmp_path, "BAAI/bge-m3", "def456")


def test_an_absent_adapter_says_which_run_produces_it(tmp_path):
    with pytest.raises(AdapterMismatch, match="train_encoder"):
        read_adapter(tmp_path / "absent", "BAAI/bge-m3", "abc123")


# --- Where the check actually runs -------------------------------------------


def test_loading_an_adapter_checks_it_before_any_weights_are_fetched(tmp_path):
    """A 2 GB download is the wrong place to discover a mismatched adapter."""
    from llms4subjects.config import EncoderConfig
    from llms4subjects.stages import encoders

    (tmp_path / ADAPTER_MANIFEST).write_text(
        json.dumps(
            adapter_manifest(
                base="Alibaba-NLP/gte-multilingual-base",
                revision="abc123",
                data_revision="rev1",
                negatives="key1",
                hyperparameters={},
                losses=[],
                seconds=0.0,
                host="nlp2",
            )
        )
    )

    with pytest.raises(AdapterMismatch, match="gte-multilingual-base"):
        encoders.load(
            EncoderConfig(name="BAAI/bge-m3", adapter=str(tmp_path)), "cpu"
        )

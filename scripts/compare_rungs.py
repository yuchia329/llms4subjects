"""Compare two encoder screens taken at different index sizes.

    python scripts/screen_encoders.py configs/rung1.yaml configs/rung1-*.yaml \\
        --json reference/screens/rung1.json
    python scripts/screen_encoders.py configs/rung2.yaml configs/rung2-*.yaml \\
        --json reference/screens/rung2.json

    python scripts/compare_rungs.py \\
        reference/screens/rung1.json reference/screens/rung2.json

Rung 2 of the experiment ladder (ticket 10) asks one question of the ladder
itself: **does the encoder ranking change when the index grows fourfold?** If it
holds, the cheap screen at 8,000 documents generalises and the fine-tuning
shortlist rests on evidence. If it flips, that is a finding of its own and the
shortlist has to be reconsidered before any GPU time is spent.

The comparison is arithmetic over two JSON documents rather than a third pass
over the models, which is why `screen_encoders.py --json` exists: the screens
cost hours of Apple Silicon each, and the claim they support has to be
re-derivable in milliseconds, months later, by someone reading the writeup.

**Only index size may move.** Every screen row carries the configuration
sections its screen held equal, and this script refuses a pair that disagrees on
any of them — a rung 2 with retuned fusion weights would measure the tuning and
report it as a scaling result. It also refuses screens over different encoder
sets, over different splits, and two screens at the same index size, none of
which are two rungs of a ladder.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms4subjects.stages.evaluator import BANDS  # noqa: E402

# The screen document this script reads. Versioned because the file outlives the
# run that wrote it: a screen from a month ago has to be either readable or
# refused, never silently misread.
SCHEMA = "encoder-screen/1"

# Below three encoders a rank correlation is not a measurement: two points can
# only agree or reverse, so rho is ±1 whatever the scores did.
MINIMUM_FOR_CORRELATION = 3


class NotAComparison(ValueError):
    """Screens whose difference is not index size, so the answer would not be."""


@dataclass(frozen=True)
class Row:
    """One encoder's result inside one screen."""

    encoder: str
    config: str
    dimensions: int
    parameters: int | None
    load_seconds: float
    retrieve_seconds: float
    wrote: int
    micro: Mapping[int, float]
    official_macro: Mapping[int, float]
    bands: Mapping[str, Mapping[int, float]]
    per_retriever: Mapping[str, Mapping[int, float]]
    # What the screen that produced this row held equal across its encoders,
    # carried per row rather than per screen so that a row out of step with its
    # own screen is as visible as a rung out of step with the other rung.
    held_equal: Mapping[str, object]

    @property
    def seconds(self) -> float:
        return self.load_seconds + self.retrieve_seconds

    @property
    def cache(self) -> str:
        return f"computed {self.wrote}" if self.wrote else "warm"


@dataclass(frozen=True)
class Screen:
    """One whole screen: several encoders at one index size."""

    path: Path
    split: str
    records: int
    data_revision: str
    selection_k: int
    device: str
    documents: int
    corpus: int
    corpora: tuple[str, ...]
    stratify: bool
    seed: int
    rows: tuple[Row, ...]

    @property
    def encoders(self) -> frozenset[str]:
        return frozenset(row.encoder for row in self.rows)

    @property
    def size(self) -> str:
        """The index size as a table heading: `8,000`, `32,043`."""
        return f"{self.documents:,}"

    def row(self, encoder: str) -> Row:
        for row in self.rows:
            if row.encoder == encoder:
                return row
        raise KeyError(encoder)


def load_screen(path: str | Path) -> Screen:
    """Read one screen document, or say why it is not one."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise NotAComparison(f"cannot read screen {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise NotAComparison(f"{path} is not JSON: {error}") from None

    schema = payload.get("schema")
    if schema != SCHEMA:
        raise NotAComparison(
            f"{path} declares schema {schema!r}, not {SCHEMA!r}; it was not "
            "written by scripts/screen_encoders.py --json"
        )

    index = payload.get("index", {})
    return Screen(
        path=path,
        split=payload["split"],
        records=payload["records"],
        data_revision=payload["data_revision"],
        selection_k=payload["selection_k"],
        device=payload.get("device", "unknown"),
        documents=index["documents"],
        corpus=index["corpus"],
        corpora=tuple(index.get("corpora", ())),
        stratify=index.get("stratify", False),
        seed=index.get("seed", 0),
        rows=tuple(_row(entry) for entry in payload["rows"]),
    )


def _row(entry: Mapping[str, object]) -> Row:
    return Row(
        encoder=entry["encoder"],
        config=entry.get("config", ""),
        dimensions=entry["dimensions"],
        parameters=entry.get("parameters"),
        load_seconds=entry["load_seconds"],
        retrieve_seconds=entry["retrieve_seconds"],
        wrote=entry.get("wrote", 0),
        micro=_at_k(entry["micro"]),
        official_macro=_at_k(entry["official_macro"]),
        bands={band: _at_k(values) for band, values in entry["bands"].items()},
        per_retriever={
            name: _at_k(values) for name, values in entry["per_retriever"].items()
        },
        held_equal=entry.get("held_equal", {}),
    )


def _at_k(values: Mapping[str, float]) -> dict[int, float]:
    """JSON object keys are strings; every k in this project is an integer."""
    return {int(k): float(value) for k, value in values.items()}


def by_index_size(screens: Sequence[Screen]) -> list[Screen]:
    """Smallest index first, never the order the paths were typed in.

    Every table below reads left to right as the index grows, and a comparison
    whose columns moved with the shell history would be a different document
    each time it ran.
    """
    return sorted(screens, key=lambda screen: screen.documents)


def check(screens: Sequence[Screen]) -> None:
    """Refuse anything that would make the two rankings incomparable."""
    if len(screens) < 2:
        raise NotAComparison(
            f"comparing rungs needs at least two screens; got {len(screens)}"
        )

    thin = [screen for screen in screens if len(screen.rows) < 2]
    if thin:
        raise NotAComparison(
            f"{thin[0].path} screens one encoder, so it has no ranking to compare"
        )

    sizes = [screen.documents for screen in screens]
    repeated = sorted({size for size in sizes if sizes.count(size) > 1})
    if repeated:
        raise NotAComparison(
            f"two screens at the same index size ({repeated[0]} documents); the "
            "rungs of the ladder differ in index size, so these are one rung"
        )

    splits = {screen.split for screen in screens}
    if len(splits) > 1:
        raise NotAComparison(
            f"the screens score different splits ({', '.join(sorted(splits))}); "
            "a ranking is only comparable within one split"
        )

    common = set.intersection(*(set(screen.encoders) for screen in screens))
    missing = sorted(set().union(*(screen.encoders for screen in screens)) - common)
    if missing:
        raise NotAComparison(
            f"{', '.join(missing)} screened at only some index sizes; the "
            "comparison is one ranking against another over the same encoders"
        )

    counts = {screen.records for screen in screens}
    if len(counts) > 1:
        raise NotAComparison(
            f"the screens score different numbers of records "
            f"({', '.join(map(str, sorted(counts)))}); a screen run under "
            "--limit is not comparable to a whole split"
        )

    corpora = {screen.corpora for screen in screens}
    if len(corpora) > 1:
        described = " against ".join(
            ", ".join(names) or "nothing" for names in sorted(corpora)
        )
        raise NotAComparison(
            f"the screens index different corpora ({described}); index size is "
            "the variable between rungs, and the corpus it is drawn from is not"
        )

    revisions = {screen.data_revision for screen in screens}
    if len(revisions) > 1:
        raise NotAComparison(
            f"the screens were run against different dataset revisions "
            f"({', '.join(sorted(revisions))}); the gold sets may differ"
        )

    differing = differences(screens)
    if differing:
        described = "\n".join(
            f"  {section}: " + " against ".join(sorted(set(map(str, values))))
            for section, values in sorted(differing.items())
        )
        raise NotAComparison(
            "the screens differ in more than index size, so the comparison "
            f"would report those differences as a scaling result:\n{described}"
        )


def differences(screens: Sequence[Screen]) -> dict[str, list[object]]:
    """What every row across every screen disagrees on, other than index size.

    Flat over rows rather than screen by screen: a section that moved between
    the rungs and a section that moved between two encoders inside one rung are
    the same defect — one number in the table was produced by a different
    experiment from the one beside it.
    """
    rows = [row for screen in screens for row in screen.rows]
    sections = sorted({section for row in rows for section in row.held_equal})
    return {
        section: [row.held_equal.get(section) for row in rows]
        for section in sections
        if len({json.dumps(row.held_equal.get(section), sort_keys=True) for row in rows})
        > 1
    }


def ranking(screen: Screen, k: int) -> list[str]:
    """The encoders of one screen, best first on micro Recall@k.

    Ties break by name, as in the screen itself: a tie decided by argument order
    would move with the command line rather than with the measurement.
    """
    return [
        row.encoder
        for row in sorted(screen.rows, key=lambda row: (-row.micro[k], row.encoder))
    ]


def ranks(values: Sequence[float]) -> list[float]:
    """Competition-free ranks, best (highest) first, ties sharing their average.

    Tied scores must share a rank: two encoders at the same recall to four
    decimals are not evidence of an order between them, and giving one of them
    the better rank would let the correlation report a stability that the
    measurement never showed.
    """
    order = sorted(range(len(values)), key=lambda i: -values[i])
    assigned = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for i in order[position : end + 1]:
            assigned[i] = average
        position = end + 1
    return assigned


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rho: Pearson correlation of the two rank vectors.

    Computed from average ranks rather than from the textbook sum-of-squares
    shortcut, because the shortcut is only correct when nothing is tied, and two
    encoders 0.0022 apart at rung 1 could tie at rung 2.
    """
    if len(x) != len(y):
        raise ValueError(f"{len(x)} values against {len(y)}")
    if len(x) < MINIMUM_FOR_CORRELATION:
        return None
    return _pearson(ranks(x), ranks(y))


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = (sum(a * a for a in dx) * sum(b * b for b in dy)) ** 0.5
    if not denominator:
        # Every value tied on one side: there is no order to correlate with.
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def kendall(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's tau-b: agreeing pairs against disagreeing ones, ties discounted.

    Tau-b rather than tau-a because a tie is not agreement. With four encoders
    there are six pairs, so tau moves in sixths and is the more readable of the
    two coefficients: "five of six pairs kept their order" is a sentence, where
    rho 0.8 is not.
    """
    if len(x) != len(y):
        raise ValueError(f"{len(x)} values against {len(y)}")
    if len(x) < MINIMUM_FOR_CORRELATION:
        return None

    concordant = discordant = tied_x = tied_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            a = x[i] - x[j]
            b = y[i] - y[j]
            if a == 0 and b == 0:
                continue
            if a == 0:
                tied_x += 1
            elif b == 0:
                tied_y += 1
            elif (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1

    pairs = concordant + discordant
    denominator = ((pairs + tied_x) * (pairs + tied_y)) ** 0.5
    if not denominator:
        return None
    return (concordant - discordant) / denominator


@dataclass(frozen=True)
class Verdict:
    """Whether the ranking held between the smallest and largest index."""

    k: int
    smaller: Screen
    larger: Screen
    stable: bool
    # (encoder, (rank at the smaller index, rank at the larger)) for each
    # encoder that moved, in the larger index's order.
    moves: tuple[tuple[str, tuple[int, int]], ...]
    spearman: float | None
    kendall: float | None


def held(screens: Sequence[Screen], k: int) -> Verdict:
    """Did the ranking hold between the smallest and the largest index?

    The two ends rather than every adjacent pair: the question the ladder asks
    is whether a screen at its cheapest size predicts the ranking at the size
    the decision is made at, and an intermediate rung agreeing with neither
    would not change that answer.
    """
    ordered = by_index_size(screens)
    smaller, larger = ordered[0], ordered[-1]

    small_order = ranking(smaller, k)
    large_order = ranking(larger, k)
    positions = {
        encoder: (small_order.index(encoder) + 1, large_order.index(encoder) + 1)
        for encoder in large_order
    }

    encoders = large_order
    return Verdict(
        k=k,
        smaller=smaller,
        larger=larger,
        stable=small_order == large_order,
        moves=tuple(
            (encoder, positions[encoder])
            for encoder in encoders
            if positions[encoder][0] != positions[encoder][1]
        ),
        spearman=spearman(
            [smaller.row(encoder).micro[k] for encoder in encoders],
            [larger.row(encoder).micro[k] for encoder in encoders],
        ),
        kendall=kendall(
            [smaller.row(encoder).micro[k] for encoder in encoders],
            [larger.row(encoder).micro[k] for encoder in encoders],
        ),
    )


def shortlist(screens: Sequence[Screen], k: int, count: int = 2) -> tuple[str, ...]:
    """The encoders carried to fine-tuning: the top `count` at the largest index.

    The largest index and not the average of the rungs. Rung 2 is the closest
    measurement available to rung 3's conditions, so where the rungs disagree it
    is the one that decides; the disagreement itself is the verdict's business,
    reported above this list rather than smoothed into it.
    """
    return tuple(ranking(by_index_size(screens)[-1], k)[:count])


def render_ranking(screens: Sequence[Screen], k: int) -> str:
    """The deliverable: both rankings side by side, ordered by the larger index."""
    ordered = by_index_size(screens)
    largest = ordered[-1]
    lines = [
        f"### The encoder ranking at each index size (dev micro R@{k})\n",
        "| encoder | dim | "
        + " | ".join(f"{screen.size} docs" for screen in ordered)
        + " | "
        + " | ".join(f"rank @ {screen.size}" for screen in ordered)
        + " | Δ |",
        "|---|---:" + "|---:" * len(ordered) + "|---:" * len(ordered) + "|---:|",
    ]
    for encoder in ranking(largest, k):
        scores = [screen.row(encoder).micro[k] for screen in ordered]
        places = [ranking(screen, k).index(encoder) + 1 for screen in ordered]
        lines.append(
            f"| `{encoder}` | {largest.row(encoder).dimensions} | "
            + " | ".join(f"{score:.4f}" for score in scores)
            + " | "
            + " | ".join(str(place) for place in places)
            + f" | {scores[-1] - scores[0]:+.4f} |"
        )
    return "\n".join(lines)


def render_bands(screens: Sequence[Screen], k: int) -> str:
    """Every encoder's frozen-band breakdown at every index size.

    One row per encoder per rung rather than a column per band per rung, so tail
    behaviour is read down a column as the index grows — which is the comparison
    the ticket asks for, and the one a wide table hides.
    """
    ordered = by_index_size(screens)
    largest = ordered[-1]
    bands = [
        band for band in BANDS if any(band in row.bands for row in largest.rows)
    ]
    lines = [
        f"### By frequency band, at each index size (micro R@{k})\n",
        "| encoder | index | " + " | ".join(bands) + " |",
        "|---|---:" + "|---:" * len(bands) + "|",
    ]
    for encoder in ranking(largest, k):
        for screen in ordered:
            row = screen.row(encoder)
            cells = " | ".join(
                f"{row.bands[band][k]:.4f}" if band in row.bands else "—"
                for band in bands
            )
            lines.append(f"| `{encoder}` | {screen.size} | {cells} |")
    return "\n".join(lines)


# The retrievers whose scores cannot depend on index size. The dense retriever
# scores the 79,427-entry label tower and BM25 reads the labels' own strings;
# neither looks at an indexed document, so both are the control on this whole
# comparison — a column of theirs that moved between the rungs would mean
# something other than the index did.
INDEX_INDEPENDENT = ("dense", "lexical")


def control_drift(
    screens: Sequence[Screen], k: int
) -> dict[str, tuple[tuple[str, tuple[float, ...]], ...]]:
    """Which index-independent retrievers moved between the rungs, and where.

    A retriever missing from one of the rungs is not drift: a rung that ran kNN
    alone has no dense column to disagree with. Only a retriever measured at
    every index size for one encoder can be a control for that encoder.
    """
    ordered = by_index_size(screens)
    drifted: dict[str, list[tuple[str, tuple[float, ...]]]] = {}
    for name in INDEX_INDEPENDENT:
        for encoder in sorted(ordered[-1].encoders):
            values = [
                screen.row(encoder).per_retriever[name][k]
                for screen in ordered
                if name in screen.row(encoder).per_retriever
                and k in screen.row(encoder).per_retriever[name]
            ]
            if len(values) < len(ordered) or len(set(values)) == 1:
                continue
            drifted.setdefault(name, []).append((encoder, tuple(values)))
    return {name: tuple(rows) for name, rows in drifted.items()}


def render_per_retriever(screens: Sequence[Screen], k: int) -> str:
    """Each retriever alone at each index size, so a gain is attributable.

    The point of the table is that only one of its columns can move. The index
    is what the kNN retriever searches, so growing it fourfold is a change to
    that retriever and to nothing else; the dense and lexical columns are
    arithmetic over the same label tower at both rungs and must be identical to
    four decimals. When they are, every difference in the fused number is
    localised to the neighbour retriever. When they are not, the comparison has
    a variable nobody declared, and that is worth more than the table.
    """
    ordered = by_index_size(screens)
    largest = ordered[-1]
    names = sorted(
        {name for screen in screens for row in screen.rows for name in row.per_retriever}
    )
    lines = [
        f"### Each retriever alone, at each index size (micro R@{k})\n",
        "| encoder | index | " + " | ".join(names) + " | fused |",
        "|---|---:" + "|---:" * (len(names) + 1) + "|",
    ]
    for encoder in ranking(largest, k):
        for screen in ordered:
            row = screen.row(encoder)
            cells = " | ".join(
                f"{row.per_retriever[name][k]:.4f}"
                if name in row.per_retriever and k in row.per_retriever[name]
                else "—"
                for name in names
            )
            lines.append(
                f"| `{encoder}` | {screen.size} | {cells} | {row.micro[k]:.4f} |"
            )

    drift = control_drift(screens, k)
    controls = [name for name in INDEX_INDEPENDENT if name in names]
    if not controls:
        return "\n".join(lines)
    if not drift:
        lines.append(
            f"\nThe {' and '.join(controls)} columns are the control and they "
            "held: neither retriever reads an indexed document, so every "
            "difference above belongs to the neighbour retriever."
        )
    else:
        moved = "; ".join(
            f"{name} — "
            + ", ".join(
                f"`{encoder}` "
                + " → ".join(f"{value:.4f}" for value in values)
                for encoder, values in rows
            )
            for name, rows in sorted(drift.items())
        )
        lines.append(
            f"\n**The control moved.** {moved}. Neither retriever reads an "
            "indexed document, so a column of theirs that changed means the "
            "rungs differ in something other than index size."
        )
    return "\n".join(lines)


def _why_no_correlation(verdict: Verdict) -> str:
    """Both coefficients came back undefined. Say which of the two reasons it was.

    Too few encoders is one; every encoder scoring the same at one index size is
    the other, and reporting that as "too few" would misdescribe a screen where
    something had gone wrong enough to tie four models to four decimals.
    """
    if len(verdict.larger.rows) < MINIMUM_FOR_CORRELATION:
        return (
            f"no rank correlation: {len(verdict.larger.rows)} encoders is below "
            f"the {MINIMUM_FOR_CORRELATION} a coefficient needs to mean anything"
        )
    return (
        "no rank correlation: at one of the index sizes every encoder scored the "
        f"same at R@{verdict.k}, so there is no order to correlate with"
    )


def render_verdict(verdict: Verdict) -> str:
    """The answer, in words, because a table does not state a conclusion."""
    correlation = " and ".join(
        part
        for part in (
            f"Spearman ρ {verdict.spearman:.3f}"
            if verdict.spearman is not None
            else "",
            f"Kendall τ-b {verdict.kendall:.3f}" if verdict.kendall is not None else "",
        )
        if part
    ) or _why_no_correlation(verdict)

    scale = (
        f"{verdict.smaller.documents:,} documents to {verdict.larger.documents:,} "
        f"({verdict.larger.documents / max(verdict.smaller.documents, 1):.1f}×)"
    )

    if verdict.stable:
        return (
            f"**The ranking held.** Every encoder keeps its place from "
            f"{scale}, so the cheap screen generalises and the fine-tuning "
            f"shortlist rests on evidence rather than on the index it was "
            f"measured at ({correlation} on dev micro R@{verdict.k})."
        )

    moved = "; ".join(
        f"`{encoder}` {before} → {after}" for encoder, (before, after) in verdict.moves
    )
    return (
        f"**The ranking flipped.** Growing the index from {scale} moved "
        f"{len(verdict.moves)} of {len(verdict.larger.rows)} encoders "
        f"({moved}), so a screen at the smaller index does not predict the "
        f"larger one and the shortlist is decided at the larger "
        f"({correlation} on dev micro R@{verdict.k})."
    )


def render_shortlist(screens: Sequence[Screen], k: int, count: int = 2) -> str:
    """Which encoders go to rung 3, and the number each one is chosen on."""
    ordered = by_index_size(screens)
    largest = ordered[-1]
    order = ranking(largest, k)
    chosen = shortlist(screens, k, count)
    if not chosen:
        return "### Shortlisted for fine-tuning: nothing was asked for\n"
    runner_up = order[count] if len(order) > count else None

    lines = [
        f"### Shortlisted for fine-tuning: {', '.join(f'`{name}`' for name in chosen)}\n"
    ]
    for place, encoder in enumerate(chosen, start=1):
        row = largest.row(encoder)
        history = ", ".join(
            f"{screen.row(encoder).micro[k]:.4f} at {screen.size}"
            for screen in ordered
        )
        zero = row.bands.get("zero", {}).get(k)
        tail = row.bands.get("tail", {}).get(k)
        bands = (
            f" Tail {tail:.4f}, zero-shot {zero:.4f}."
            if tail is not None and zero is not None
            else ""
        )
        lines.append(
            f"{place}. **`{encoder}`** — {history} (micro R@{k}), "
            f"{row.dimensions} dimensions.{bands}"
        )
    if runner_up is not None:
        margin = largest.row(chosen[-1]).micro[k] - largest.row(runner_up).micro[k]
        lines.append(
            f"\nThe margin over the first encoder left out, `{runner_up}`, is "
            f"{margin:+.4f} at {largest.size} documents."
        )
    return "\n".join(lines)


def render_cost(screens: Sequence[Screen], k: int) -> str:
    """What each rung cost per encoder, with the cache state behind the figure."""
    ordered = by_index_size(screens)
    largest = ordered[-1]
    lines = [
        "### Wall clock, per encoder and index size\n",
        "| encoder | index | load | stage one | total | cache |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for encoder in ranking(largest, k):
        for screen in ordered:
            row = screen.row(encoder)
            lines.append(
                f"| `{encoder}` | {screen.size} | {row.load_seconds:.1f}s | "
                f"{row.retrieve_seconds:.1f}s | {row.seconds:.1f}s | {row.cache} |"
            )
    return "\n".join(lines)


def _at_least_one(value: str) -> int:
    """A shortlist of nothing is not a shortlist, and would print half a report."""
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError(
            f"--shortlist names the encoders carried forward, so it needs at "
            f"least one; got {count}"
        )
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "screens", nargs="+", help="screen documents from screen_encoders.py --json"
    )
    parser.add_argument(
        "--k",
        type=int,
        help="the selection metric's k; defaults to the screens' own",
    )
    parser.add_argument(
        "--shortlist",
        type=_at_least_one,
        default=2,
        help="how many encoders to carry to fine-tuning (default 2)",
    )
    args = parser.parse_args(argv)

    try:
        screens = [load_screen(path) for path in args.screens]
        check(screens)
    except NotAComparison as error:
        print(error)
        return 1

    ks = {screen.selection_k for screen in screens}
    if args.k is None and len(ks) > 1:
        print(
            f"the screens select on different k ({', '.join(map(str, sorted(ks)))}); "
            "pass --k to say which this comparison is read on"
        )
        return 1
    k = args.k if args.k is not None else ks.pop()

    missing = [
        f"{screen.path}: `{row.encoder}` has no R@{k}"
        for screen in screens
        for row in screen.rows
        if k not in row.micro
    ]
    if missing:
        print("\n".join(missing))
        return 1

    ordered = by_index_size(screens)
    print(
        f"comparing {len(ordered)} screens of {len(ordered[-1].rows)} encoders "
        f"over {ordered[-1].records} {ordered[-1].split} records"
    )
    print(f"data revision: {ordered[-1].data_revision}")
    for screen in ordered:
        print(
            f"  {screen.size} of {screen.corpus:,} documents from "
            f"{', '.join(screen.corpora)}, stratify={screen.stratify}, "
            f"seed={screen.seed} — {screen.path}"
        )
    print()
    print(render_ranking(ordered, k))
    print()
    print(render_verdict(held(ordered, k)))
    print()
    print(render_per_retriever(ordered, k))
    print()
    print(render_bands(ordered, k))
    print()
    print(render_shortlist(ordered, k, args.shortlist))
    print()
    print(render_cost(ordered, k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

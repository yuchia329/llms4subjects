"""The named artifacts stages pass between each other.

These types are the whole vocabulary of the pipeline: a stage takes them as
arguments and returns them, so that no stage has to know where the data came
from or where the next one will put it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Codes are the GND identifiers as they appear in the data, e.g. "gnd:4043744-9".
Code = str

# The submission format takes exactly this many ranked codes per record, and has
# no representation for fewer. Everything that reduces a ranking reads it here.
CODES_PER_RECORD = 50


@dataclass(frozen=True)
class Record:
    """One TIBKAT record, carrying its official id so splits stay traceable.

    `subjects` is present for indexed training records and for gold evaluation,
    and must never reach candidate generation for the record being predicted.
    """

    id: str
    type: str
    lang: str
    title: str
    abstract: str
    subjects: tuple[Code, ...] = ()

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


@dataclass(frozen=True)
class VocabularyEntry:
    """One GND subject heading, as released in the shared task's vocabulary."""

    code: Code
    name: str
    classification_name: str = ""
    classification_number: str = ""
    alternate_names: tuple[str, ...] = ()
    related_subjects: tuple[str, ...] = ()
    definition: str = ""


@dataclass(frozen=True)
class LabelText:
    """A vocabulary entry rendered for an encoder, with its rendering flags."""

    code: Code
    text: str


@dataclass(frozen=True)
class Candidate:
    """One proposed code for one record, with the provenance of its score.

    `sources` maps retriever name to that retriever's rank for this candidate,
    so a final prediction can be attributed to the component that found it.
    """

    code: Code
    score: float
    sources: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateList:
    """The candidates for one record, in non-increasing score order."""

    record_id: str
    candidates: tuple[Candidate, ...]

    @property
    def codes(self) -> tuple[Code, ...]:
        return tuple(candidate.code for candidate in self.candidates)

    def top(self, k: int) -> "CandidateList":
        return CandidateList(self.record_id, self.candidates[:k])

    def as_prediction(self, k: int) -> "Prediction":
        """The top k codes as the submission writer's contract."""
        return Prediction(self.record_id, self.codes[:k])


@dataclass(frozen=True)
class Prediction:
    """The final contract: exactly 50 ranked codes for one record."""

    record_id: str
    codes: tuple[Code, ...]

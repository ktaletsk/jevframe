"""Frozen request definitions and deterministic native result columns."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeError,
)

from ._types import Question

# Allow floating-point/API rounding, without repairing or renormalizing values.
PROBABILITY_TOLERANCE = 1e-6


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class Column:
    name: str
    kind: Literal["float", "int", "str"]


@dataclass(frozen=True)
class QuestionPlan:
    questions: dict[str, Question]
    columns: tuple[Column, ...]
    serialized: str


def pack_rows(values: list[tuple[Any, ...]], plan: QuestionPlan) -> list[dict[str, Any] | None]:
    """Pack validated scalar fields, keeping failed/skipped rows wholly missing.

    Build fresh dictionaries so callers cannot mutate other rows or cached values.
    Field names and order exactly match the expanded column schema.
    """
    names = [column.name for column in plan.columns]
    return [
        None if all(value is None for value in row) else dict(zip(names, row, strict=True))
        for row in values
    ]


def make_plan(questions: Mapping[str, Question], *, prefix: bool) -> QuestionPlan:
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a nonempty mapping of names to SDK question objects")
    snapshots: dict[str, Question] = {}
    columns: list[Column] = []
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Question names must be nonempty strings")
        if not isinstance(question, (Noul, Choice, Score)):
            raise TypeError("questions values must be official SDK Noul, Choice, or Score objects")
        cls = (
            Noul
            if isinstance(question, Noul)
            else Choice
            if isinstance(question, Choice)
            else Score
        )
        # These SDK models are mutable; validation of a dumped copy prevents drift.
        snapshot = cls.model_validate(question.model_dump(mode="json"))
        snapshots[name] = snapshot
        fields: list[tuple[str, Literal["float", "int", "str"]]]
        if isinstance(snapshot, Noul):
            fields = [("probability", "float")]
        elif isinstance(snapshot, Choice):
            if not 1 <= len(snapshot.criteria) <= 255:
                raise ValueError("Choice requires 1 to 255 options")
            fields = [("label", "str"), ("confidence", "float")]
            fields += [(f"p__{label}", "float") for label in snapshot.criteria]
        else:
            if not 2 <= len(snapshot.criteria) <= 10:
                raise ValueError("Score requires 2 to 10 ordered levels")
            fields = [
                ("level", "int"),
                ("label", "str"),
                ("score", "float"),
                ("confidence", "float"),
            ]
            fields += [(f"p__{i}", "float") for i in range(len(snapshot.criteria))]
        columns.extend(
            Column(f"{name}__{field}" if prefix else field, kind) for field, kind in fields
        )
    names = [column.name for column in columns]
    if len(set(names)) != len(names):
        raise ValueError("Generated result column names collide; rename questions or choice labels")
    serialized = encode({name: q.model_dump(mode="json") for name, q in snapshots.items()})
    return QuestionPlan(snapshots, tuple(columns), serialized)


def choice_question(instructions: str, choices: Sequence[str] | Mapping[str, Any]) -> Choice:
    if isinstance(choices, Mapping):
        criteria = dict(choices)
    elif isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        if any(not isinstance(label, str) for label in choices):
            raise TypeError("Choice labels must be strings")
        if len(set(choices)) != len(choices):
            raise ValueError("Choice labels must be unique")
        criteria = dict.fromkeys(choices)
    else:
        raise TypeError("choices must be a sequence of labels or a mapping of descriptions")
    return Choice(instructions=instructions, criteria=criteria)


def probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric probability")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Probability must be finite and between zero and one")
    return number


def distribution(values: Mapping[Any, float], keys: Sequence[Any]) -> list[float]:
    if set(values) != set(keys):
        raise ValueError("Probability distribution keys differ from the requested criteria")
    result = [probability(values[key]) for key in keys]
    if not math.isclose(math.fsum(result), 1, rel_tol=0, abs_tol=PROBABILITY_TOLERANCE):
        raise ValueError("Probability distribution must sum to one")
    return result


def flatten(response: SystemOneResponse, plan: QuestionPlan) -> tuple[Any, ...]:
    if not isinstance(response, SystemOneResponse):
        raise TypeError("Expected an official SDK SystemOneResponse")
    # The SDK omits unknown answer types. Check raw names too, so an unexpected
    # future answer cannot disappear before our exact question-set validation.
    try:
        raw_names = response.raw_http_response.json().get("answers", {})
    except TypeSafeError:
        raw_names = response.answers  # SDK models constructed directly, e.g. in tests.
    if not isinstance(raw_names, dict) or set(raw_names) != set(plan.questions):
        raise ValueError("Response question names differ from the request")
    if set(response.answers) != set(plan.questions):
        raise ValueError("Response question names differ from the request")
    result: list[Any] = []
    for name, question in plan.questions.items():
        answer = response.answers[name]
        if isinstance(question, Noul):
            if not isinstance(answer, NoulAnswer):
                raise ValueError("Expected a Noul answer")
            result.append(probability(answer.noul))
        elif isinstance(question, Choice):
            if not isinstance(answer, ChoiceAnswer):
                raise ValueError("Expected a Choice answer")
            probabilities = distribution(answer.probabilities, list(question.criteria))
            if answer.choice not in question.criteria:
                raise ValueError("Selected choice is outside the requested criteria")
            result.extend([answer.choice, probability(answer.confidence), *probabilities])
        else:
            if not isinstance(answer, ScoreAnswer):
                raise ValueError("Expected a Score answer")
            levels = list(question.criteria)
            probabilities = distribution(answer.probabilities, list(range(len(levels))))
            if answer.legend != dict(enumerate(levels)):
                raise ValueError("Score legend differs from the requested levels")
            score = answer.score
            expected = math.fsum(i * p for i, p in enumerate(probabilities))
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= len(levels) - 1
                or not math.isclose(
                    score, expected, rel_tol=0, abs_tol=PROBABILITY_TOLERANCE * len(levels)
                )
            ):
                raise ValueError("Score is not the expected zero-based level")
            level = max(range(len(levels)), key=probabilities.__getitem__)
            label = levels[level] if isinstance(levels[level], str) else encode(levels[level])
            result.extend(
                [level, label, float(score), probability(answer.confidence), *probabilities]
            )
    return tuple(result)

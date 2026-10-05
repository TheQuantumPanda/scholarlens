#!/usr/bin/env python3
"""Report recorded Phase 6A observations without running retrieval or providers."""

from __future__ import annotations

import argparse
from pathlib import Path

from scholarlens.evaluation import aggregate, diagnose, load_fixture, load_runs


DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/phase6a_end_to_end_reference.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Recorded JSON run file(s); each may contain one run or an array")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()
    fixture, fixture_hash = load_fixture(args.fixture)
    references = {q.question_id: q for q in fixture.questions}
    runs = [run for path in args.runs for run in load_runs(path)]
    pairs = []
    for run in runs:
        if run.question_id not in references:
            parser.error(f"Unknown question ID: {run.question_id}")
        if (run.metadata.evaluation_fixture_hash is not None
                and run.metadata.evaluation_fixture_hash != fixture_hash):
            parser.error(f"Fixture hash mismatch for run {run.metadata.run_id}")
        pairs.append((references[run.question_id], run))
    for ref, run in pairs:
        result = diagnose(ref, run)
        print(f"{run.metadata.run_id} / {run.question_id} [{ref.answerability.value}]")
        print(f"  retrieval: {result.retrieval_diagnosis}")
        print(f"  generation: {result.generation_diagnosis}")
        print(f"  verification: {result.verification_diagnosis}")
        print(f"  outcome: {result.primary_outcome.value}")
    for group, metrics in aggregate(pairs).items():
        print(f"\n{group}:")
        for name, metric in metrics.items():
            percent = "undefined" if metric.percentage is None else f"{metric.percentage:.1f}%"
            print(f"  {name}: {metric.numerator}/{metric.denominator} ({percent})")


if __name__ == "__main__":
    main()

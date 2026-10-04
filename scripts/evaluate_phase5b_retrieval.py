#!/usr/bin/env python3
"""Offline development evaluation for Phase 5B retrieval strategies.

This script is deliberately not imported by the Streamlit application. It reads
local PDFs and a manually labeled fixture, builds an ephemeral BGE index, and
never calls a generation provider.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/phase5b_rag_retrieval_devset.json"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
RRF_CONSTANT = 60


@dataclass(frozen=True)
class Candidate:
    paper_key: str
    result: Any
    rank: int
    distance: float
    sources: tuple[tuple[str, int, float], ...]
    cross_score: float | None = None
    cross_rank: int | None = None
    fusion_score: float | None = None

    @property
    def chunk_key(self) -> tuple[str, str]:
        return self.paper_key, self.result.chunk_id.split(":", 1)[1]


def _load_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != 1:
        raise ValueError("Unsupported development fixture schema")
    return fixture


def _filtered(candidates: list[Candidate], junk_reasons: dict[tuple[str, str], str | None]) -> list[Candidate]:
    return [c for c in candidates if junk_reasons.get(c.chunk_key) is None]


def _coverage_select(
    question_text: str,
    paper_order: list[str],
    paper_ids: dict[str, str],
    candidates: dict[str, list[Candidate]],
    model: str,
    budget_tokens: int,
    max_chunks: int,
    scoring: str,
) -> tuple[list[Candidate], int]:
    """Choose a feasible coverage seed, then greedily admit ranked additions."""
    from scholarlens.cross_paper import (
        CrossPaperPool,
        PaperCandidates,
        build_cross_paper_messages,
        estimate_cross_paper_tokens,
    )

    flattened = [candidate for key in paper_order for candidate in candidates[key]]

    def cost(selected: list[Candidate]) -> int:
        records_by_key: dict[str, list[Any]] = {key: [] for key in paper_order}
        for item in flattened:
            records_by_key[item.paper_key].append(item.result)
        papers = tuple(PaperCandidates(paper_ids[key], tuple(records_by_key[key])) for key in paper_order)
        ordered = sorted(
            selected,
            key=lambda c: (paper_order.index(c.paper_key), c.rank, c.distance, c.result.chunk_id),
        )
        pool = CrossPaperPool(question_text, papers, tuple(c.result for c in ordered))
        return estimate_cross_paper_tokens(build_cross_paper_messages(pool), model)

    seeds: list[tuple[tuple[Any, ...], list[Candidate]]] = []
    # At most one passage per paper in the first coverage round.
    import itertools

    for count in range(1, len(paper_order) + 1):
        for selected_keys in itertools.combinations(paper_order, count):
            per_paper = [candidates[key] for key in selected_keys]
            if any(not values for values in per_paper):
                continue
            for combo in itertools.product(*per_paper):
                selected = list(combo)
                estimated = cost(selected)
                if estimated > budget_tokens:
                    continue
                rank_sum = sum(candidate.rank for candidate in selected)
                distance_sum = sum(candidate.distance for candidate in selected)
                if scoring == "cross":
                    score = -sum(candidate.cross_score or 0.0 for candidate in selected)
                elif scoring == "fusion":
                    score = -sum(candidate.fusion_score or 0.0 for candidate in selected)
                else:
                    score = 0.0
                seeds.append(((-count, score, rank_sum, distance_sum, estimated), selected))

    selected = min(seeds, key=lambda item: item[0])[1] if seeds else []

    def admission_key(candidate: Candidate) -> tuple[Any, ...]:
        incremental_cost = cost([*selected, candidate])
        if scoring == "cross":
            primary: Any = -(candidate.cross_score or 0.0)
        elif scoring == "fusion":
            primary = -(candidate.fusion_score or 0.0)
        else:
            primary = candidate.rank
        return (primary, candidate.rank, candidate.distance, incremental_cost, candidate.result.chunk_id)

    remaining = sorted(
        (candidate for candidate in flattened if candidate not in selected),
        key=admission_key,
    )
    for candidate in remaining:
        if len(selected) >= max_chunks:
            break
        if cost([*selected, candidate]) <= budget_tokens:
            selected.append(candidate)
    return selected, cost(selected)


def _current_select(
    question_text: str,
    paper_order: list[str],
    paper_ids: dict[str, str],
    candidates: dict[str, list[Candidate]],
    model: str,
    budget_tokens: int,
    max_chunks: int,
) -> tuple[list[Candidate], int]:
    """Mirror retrieve_cross_paper_evidence's rank-round/distance admission."""
    from scholarlens.cross_paper import (
        CrossPaperPool,
        PaperCandidates,
        build_cross_paper_messages,
        estimate_cross_paper_tokens,
    )

    def cost(selected: list[Candidate]) -> int:
        papers = tuple(
            PaperCandidates(paper_ids[key], tuple(c.result for c in candidates[key]))
            for key in paper_order
        )
        ordered = sorted(selected, key=lambda c: (paper_order.index(c.paper_key), c.rank, c.distance))
        pool = CrossPaperPool(question_text, papers, tuple(c.result for c in ordered))
        return estimate_cross_paper_tokens(build_cross_paper_messages(pool), model)

    priority = []
    for rank in range(1, 4):
        round_items = [c for key in paper_order for c in candidates[key] if c.rank == rank]
        round_items.sort(key=lambda c: (c.distance, paper_order.index(c.paper_key)))
        priority.extend(round_items)
    selected: list[Candidate] = []
    for candidate in priority:
        if len(selected) >= max_chunks:
            break
        if cost([*selected, candidate]) <= budget_tokens:
            selected.append(candidate)
    if max_chunks >= 2 and len({c.paper_key for c in selected}) < 2:
        pair = next(
            (
                [first, second]
                for index, first in enumerate(priority)
                for second in priority[index + 1 :]
                if first.paper_key != second.paper_key and cost([first, second]) <= budget_tokens
            ),
            None,
        )
        if pair:
            selected = pair
            for candidate in priority:
                if len(selected) >= max_chunks:
                    break
                if candidate not in selected and cost([*selected, candidate]) <= budget_tokens:
                    selected.append(candidate)
    return selected, cost(selected)


def _calculate_metrics(
    fixture_question: dict[str, Any],
    ranked: dict[str, list[Candidate]],
    selected: list[Candidate],
    estimate: int,
) -> dict[str, Any]:
    paper_keys = list(fixture_question["labels"])
    labels = fixture_question["labels"]
    judged: dict[str, set[str]] = {key: set() for key in paper_keys}
    for key in paper_keys:
        for run in fixture_question["judged_runs"]:
            judged[key].update(run["by_paper"].get(key, []))

    precision: dict[str, dict[str, float]] = {}
    useful_recall, partial_recall, any_recall, mrr = [], [], [], []
    for key in paper_keys:
        top = ranked[key][:3]
        judgments = [labels[key][c.chunk_key[1]]["label"] for c in top]
        precision[key] = {
            "useful": sum(j == "USEFUL" for j in judgments) / 3,
            "partial": sum(j == "PARTIAL" for j in judgments) / 3,
            "irrelevant": sum(j == "IRRELEVANT" for j in judgments) / 3,
        }
        all_judgments = [labels[key][chunk]["label"] for chunk in judged[key]]
        useful_total = sum(j == "USEFUL" for j in all_judgments)
        partial_total = sum(j == "PARTIAL" for j in all_judgments)
        up_total = useful_total + partial_total
        useful_n = sum(j == "USEFUL" for j in judgments)
        partial_n = sum(j == "PARTIAL" for j in judgments)
        if useful_total:
            useful_recall.append(useful_n / useful_total)
        if partial_total:
            partial_recall.append(partial_n / partial_total)
        if up_total:
            any_recall.append((useful_n + partial_n) / up_total)
        first_useful = next((i for i, value in enumerate(judgments, 1) if value == "USEFUL"), None)
        mrr.append(1 / first_useful if first_useful else 0.0)

    pool_labels = [labels[c.paper_key][c.chunk_key[1]]["label"] for c in selected]
    useful_papers = {
        c.paper_key
        for c in selected
        if labels[c.paper_key][c.chunk_key[1]]["label"] == "USEFUL"
    }
    represented_relevant = {
        key for key in paper_keys
        if any(labels[key][chunk]["label"] in {"USEFUL", "PARTIAL"} for chunk in judged[key])
    }
    represented_up = {
        c.paper_key for c in selected
        if labels[c.paper_key][c.chunk_key[1]]["label"] in {"USEFUL", "PARTIAL"}
    }
    all_useful_papers = {
        key for key in paper_keys
        if any(labels[key][chunk]["label"] == "USEFUL" for chunk in judged[key])
    }
    pool_precision = sum(value == "USEFUL" for value in pool_labels) / len(pool_labels) if pool_labels else 0.0
    pool_partial = sum(value == "PARTIAL" for value in pool_labels) / len(pool_labels) if pool_labels else 0.0
    return {
        "precision_at_3": precision,
        "recall_useful_at_3": statistics.mean(useful_recall) if useful_recall else None,
        "recall_partial_at_3": statistics.mean(partial_recall) if partial_recall else None,
        "recall_useful_partial_at_3": statistics.mean(any_recall) if any_recall else None,
        "mrr_first_useful": statistics.mean(mrr),
        "pool_useful_precision": pool_precision,
        "pool_partial_share": pool_partial,
        "pool_counts": {name: pool_labels.count(name) for name in ("USEFUL", "PARTIAL", "IRRELEVANT")},
        "useful_papers": len(useful_papers),
        "useful_paper_ids": sorted(useful_papers),
        "at_least_two_useful_papers": len(useful_papers) >= 2,
        "all_relevant_papers_up_covered": represented_relevant <= represented_up,
        "all_useful_papers_covered_usefully": all_useful_papers <= useful_papers,
        "estimate": estimate,
        "selected": [
            {
                "paper": c.paper_key,
                "chunk": c.chunk_key[1],
                "label": labels[c.paper_key][c.chunk_key[1]]["label"],
                "rank": c.rank,
            }
            for c in selected
        ],
    }


def run_evaluation(
    fixture_path: Path,
    pdf_dir: Path,
    bge_model: str,
    cross_encoder_model: str,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> dict[str, Any]:
    # Enforce offline weight lookup before Sentence Transformers is imported.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from sentence_transformers import CrossEncoder
    from scholarlens.candidate_filter import obvious_junk_reason
    from scholarlens.chunking import chunk_pages
    from scholarlens.cross_paper import CANDIDATES_PER_PAPER, CrossPaperConfig
    from scholarlens.embeddings import SentenceTransformerEmbedder
    from scholarlens.pdf import extract_pdf_pages
    from scholarlens.retrieval import SemanticRetriever

    fixture = _load_fixture(fixture_path)
    paper_order = [paper["key"] for paper in fixture["papers"]]
    paper_ids = {paper["key"]: paper["paper_id"] for paper in fixture["papers"]}
    chunks = []
    for paper in fixture["papers"]:
        path = pdf_dir / paper["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing local paper PDF: {path}")
        chunks.extend(
            chunk_pages(
                extract_pdf_pages(path.read_bytes(), paper["filename"], paper["paper_id"]),
                fixture["settings"]["chunk_words"],
                fixture["settings"]["overlap_words"],
            )
        )

    embedder = SentenceTransformerEmbedder(bge_model)
    retriever = SemanticRetriever(embedder)
    t0 = time.perf_counter()
    retriever.index(chunks)
    indexing_seconds = time.perf_counter() - t0

    question_data = fixture["questions"]
    actual: dict[str, dict[str, Any]] = {}
    retrieval_seconds: dict[str, dict[str, float]] = {}
    junk_reasons: dict[tuple[str, str], str | None] = {}

    for question in question_data:
        qid = question["id"]
        text = question["question"]
        runs = []
        retrieval_seconds[qid] = {}
        sources = [("original", text)]
        if question.get("focused_query"):
            sources.append(("focused", question["focused_query"]))
        for source, retrieval_query in sources:
            t0 = time.perf_counter()
            by_paper = {}
            for paper_key in paper_order:
                results = retriever.query_paper(
                    paper_ids[paper_key],
                    retrieval_query,
                    top_k=question["candidate_window_per_paper"],
                )
                by_paper[paper_key] = results
                for result in results:
                    local_chunk = result.chunk_id.split(":", 1)[1]
                    junk_reasons[(paper_key, local_chunk)] = obvious_junk_reason(result.text)
            runs.append({"source": source, "by_paper": by_paper})
            retrieval_seconds[qid][source] = time.perf_counter() - t0
        question["_runs"] = runs
        actual[qid] = {"runs": runs, "text": text}
        for run in question["judged_runs"]:
            actual_ids = {
                key: [result.chunk_id.split(":", 1)[1] for result in actual_run["by_paper"][key]]
                for actual_run in runs if actual_run["source"] == run["source"]
                for key in paper_order
            }
            if actual_ids != run["by_paper"]:
                # Fail closed on unseen candidates; an unjudged passage must not
                # silently become an irrelevant label in the aggregate metrics.
                raise ValueError(
                    f"Candidate window changed for {qid}/{run['source']}; refresh and review fixture labels."
                )
        for key in paper_order:
            judged = question["labels"].get(key, {})
            for run in runs:
                for result in run["by_paper"][key]:
                    local_chunk = result.chunk_id.split(":", 1)[1]
                    if local_chunk not in judged:
                        raise ValueError(f"Missing manual label for {qid}/{key}/{local_chunk}")

    ce_start = time.perf_counter()
    cross_encoder = CrossEncoder(cross_encoder_model, device=device, local_files_only=True)
    cross_load_seconds = time.perf_counter() - ce_start
    cross_scores: dict[str, dict[tuple[str, str], float]] = {}
    cross_seconds: dict[str, float] = {}
    for question in question_data:
        qid = question["id"]
        original_run = next(run for run in actual[qid]["runs"] if run["source"] == "original")
        pairs = [
            (actual[qid]["text"], result.text)
            for key in paper_order
            for result in original_run["by_paper"][key]
        ]
        t0 = time.perf_counter()
        scores = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
        cross_seconds[qid] = time.perf_counter() - t0
        cross_scores[qid] = {}
        cursor = 0
        for key in paper_order:
            for result in original_run["by_paper"][key]:
                local_chunk = result.chunk_id.split(":", 1)[1]
                cross_scores[qid][(key, local_chunk)] = float(scores[cursor])
                cursor += 1

    budget = CrossPaperConfig()
    configurations: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("A", "B", "C", "D", "E")
    }
    detail = {}
    for question in question_data:
        qid, qtext = question["id"], question["question"]
        raw_runs = actual[qid]["runs"]
        # Candidate lists are explicitly assembled per strategy so each keeps
        # its own rank semantics; all use the same page-aware source records.
        current = {}
        original_run = next(run for run in raw_runs if run["source"] == "original")
        for key in paper_order:
            current[key] = [
                Candidate(
                    key,
                    result,
                    result.rank,
                    result.distance,
                    (("original", result.rank, result.distance),),
                )
                for result in original_run["by_paper"][key][:CANDIDATES_PER_PAPER]
            ]
            current[key] = _filtered(current[key], junk_reasons)

        # B: union Top-3 from original and the existing focused query, where
        # the fixture marks that query as applicable (Q1 and query optimization).
        union_lists = {key: [] for key in paper_order}
        for key in paper_order:
            sources = [("original", original_run["by_paper"][key][:3])]
            focused_run = next((run for run in raw_runs if run["source"] == "focused"), None)
            if focused_run:
                sources.append(("focused", focused_run["by_paper"][key][:3]))
            dedup = {}
            for source, results in sources:
                for result in results:
                    local_chunk = result.chunk_id.split(":", 1)[1]
                    dedup.setdefault(local_chunk, {"result": result, "sources": []})
                    dedup[local_chunk]["sources"].append((source, result.rank, result.distance))
            for local_chunk, record in dedup.items():
                result = min(record["sources"], key=lambda v: (v[1], v[2], v[0]))
                union_lists[key].append(
                    Candidate(
                        key,
                        replace(record["result"], rank=result[1], distance=result[2]),
                        result[1],
                        result[2],
                        tuple(record["sources"]),
                    )
                )
            union_lists[key] = _filtered(union_lists[key], junk_reasons)
            union_lists[key].sort(key=lambda c: (c.rank, c.distance, c.result.chunk_id))

        broad_lists = {key: [] for key in paper_order}
        for key in paper_order:
            broad_lists[key] = [
                Candidate(key, result, result.rank, result.distance,
                          (("original", result.rank, result.distance),))
                for result in original_run["by_paper"][key]
            ]
            broad_lists[key] = _filtered(broad_lists[key], junk_reasons)

        # Rank cross-encoder outputs within each paper and create fixed-c=60 RRF.
        ce_lists, fusion_lists = {}, {}
        for key in paper_order:
            values = [
                replace(c, cross_score=cross_scores[qid][c.chunk_key])
                for c in broad_lists[key]
            ]
            ce_sorted = sorted(values, key=lambda c: (-(c.cross_score or 0.0), c.rank, c.distance))
            ce_ranks = {candidate.chunk_key: i for i, candidate in enumerate(ce_sorted, 1)}
            ce_lists[key] = ce_sorted
            fusion_lists[key] = [
                replace(
                    candidate,
                    cross_rank=ce_ranks[candidate.chunk_key],
                    fusion_score=1 / (RRF_CONSTANT + candidate.rank)
                    + 1 / (RRF_CONSTANT + ce_ranks[candidate.chunk_key]),
                )
                for candidate in values
            ]
            fusion_lists[key].sort(key=lambda c: (-(c.fusion_score or 0.0), c.rank, c.distance))

        strategy = {
            "A": (current, "bge", "current"),
            "B": (union_lists, "bge", "current"),
            "C": (broad_lists, "bge", "coverage"),
            "D": (ce_lists, "cross", "coverage"),
            "E": (fusion_lists, "fusion", "coverage"),
        }
        detail[qid] = {}
        for name, (lists, scoring, selection_mode) in strategy.items():
            if selection_mode == "current":
                selected, estimate = _current_select(
                    qtext, paper_order, paper_ids, lists, model_name,
                    budget.safe_prompt_tokens, budget.max_evidence_chunks,
                )
            else:
                selected, estimate = _coverage_select(
                    qtext, paper_order, paper_ids, lists, model_name,
                    budget.safe_prompt_tokens, budget.max_evidence_chunks,
                    scoring,
                )
            result = _calculate_metrics(question, lists, selected, estimate)
            configurations[name].append(result)
            detail[qid][name] = result

    summaries = {}
    for name, questions in configurations.items():
        config_retrieval_times = []
        for question in question_data:
            qid = question["id"]
            used_sources = ["original"]
            if name == "B" and "focused" in retrieval_seconds[qid]:
                used_sources.append("focused")
            config_retrieval_times.append(
                sum(retrieval_seconds[qid][source] for source in used_sources)
            )
        summaries[name] = {
            "precision_at_3_by_paper": {
                key: {
                    metric: statistics.mean(q["precision_at_3"][key][metric] for q in questions)
                    for metric in ("useful", "partial", "irrelevant")
                }
                for key in paper_order
            },
            "recall_useful_at_3": _mean_optional(q["recall_useful_at_3"] for q in questions),
            "recall_partial_at_3": _mean_optional(q["recall_partial_at_3"] for q in questions),
            "recall_useful_partial_at_3": _mean_optional(q["recall_useful_partial_at_3"] for q in questions),
            "mrr_first_useful": statistics.mean(q["mrr_first_useful"] for q in questions),
            "pool_useful_precision": _pool_precision(questions),
            "pool_partial_share": _pool_partial_share(questions),
            "pool_useful_count": sum(q["pool_counts"]["USEFUL"] for q in questions),
            "pool_partial_count": sum(q["pool_counts"]["PARTIAL"] for q in questions),
            "pool_irrelevant_count": sum(q["pool_counts"]["IRRELEVANT"] for q in questions),
            "mean_useful_papers": statistics.mean(q["useful_papers"] for q in questions),
            "questions_with_2_useful_papers_pct": 100 * statistics.mean(q["at_least_two_useful_papers"] for q in questions),
            "all_relevant_papers_covered_pct": 100 * statistics.mean(q["all_relevant_papers_up_covered"] for q in questions),
            "all_useful_papers_covered_usefully_pct": 100 * statistics.mean(q["all_useful_papers_covered_usefully"] for q in questions),
            "mean_request_estimate": statistics.mean(q["estimate"] for q in questions),
            "max_request_estimate": max(q["estimate"] for q in questions),
            "mean_retrieval_seconds_per_question": statistics.mean(config_retrieval_times),
            "mean_rerank_seconds_per_question": (
                statistics.mean(cross_seconds.values()) if name in {"D", "E"} else 0.0
            ),
        }
    candidate_diagnostics = {}
    for question in question_data:
        qid = question["id"]
        candidate_diagnostics[qid] = {}
        for run in actual[qid]["runs"]:
            candidate_diagnostics[qid][run["source"]] = {}
            for key in paper_order:
                candidate_diagnostics[qid][run["source"]][key] = []
                for result in run["by_paper"][key]:
                    chunk = result.chunk_id.split(":", 1)[1]
                    label = question["labels"][key][chunk]
                    candidate_diagnostics[qid][run["source"]][key].append({
                        "rank": result.rank,
                        "distance": result.distance,
                        "chunk": chunk,
                        "judgment": label["label"],
                        "label_note": label["note"],
                        "junk_reason": obvious_junk_reason(result.text),
                        "cross_encoder_score": (
                            cross_scores[qid].get((key, chunk))
                            if run["source"] == "original" else None
                        ),
                        "chars": len(result.text),
                        "text_preview": result.text[:300].replace("\n", " "),
                    })
    return {
        "fixture": str(fixture_path),
        "papers": fixture["papers"],
        "questions": [q["id"] for q in question_data],
        "settings": fixture["settings"],
        "definitions": {
            "judged_recall_universe": "Unique labeled chunks in the fixture's original and applicable focused Top-6 windows; not whole-paper recall.",
            "precision_at_3": "USEFUL/PARTIAL/IRRELEVANT counts divided by 3; filtered slots remain empty rather than being backfilled.",
            "useful_paper_coverage": "A paper is useful-covered only if a USEFUL labeled passage survives the final pool.",
            "relevant_paper_coverage": "Relevant paper means at least one USEFUL or PARTIAL label in the judged window; covered if final pool retains USEFUL or PARTIAL from it.",
            "rank_fusion": f"Reciprocal-rank fusion with fixed c={RRF_CONSTANT}; no question-specific tuning.",
        },
        "timing": {
            "index_seconds": indexing_seconds,
            "retrieval_seconds_by_question_and_source": retrieval_seconds,
            "cross_encoder_load_seconds": cross_load_seconds,
            "mean_rerank_seconds_per_question": statistics.mean(cross_seconds.values()),
            "rerank_seconds_by_question": cross_seconds,
        },
        "aggregate": summaries,
        "per_question": detail,
        "candidate_diagnostics": candidate_diagnostics,
    }


def _mean_optional(values: Any) -> float | None:
    observed = [value for value in values if value is not None]
    return statistics.mean(observed) if observed else None


def _pool_precision(questions: list[dict[str, Any]]) -> float:
    total = sum(q["pool_counts"][name] for q in questions for name in ("USEFUL", "PARTIAL", "IRRELEVANT"))
    return sum(q["pool_counts"]["USEFUL"] for q in questions) / total if total else 0.0


def _pool_partial_share(questions: list[dict[str, Any]]) -> float:
    total = sum(q["pool_counts"][name] for q in questions for name in ("USEFUL", "PARTIAL", "IRRELEVANT"))
    return sum(q["pool_counts"]["PARTIAL"] for q in questions) / total if total else 0.0


def _render_report(result: dict[str, Any]) -> str:
    paper_keys = [paper["key"] for paper in result["papers"]]
    lines = [
        "# Phase 5B offline retrieval development evaluation",
        "",
        f"Questions: {', '.join(result['questions'])}. Judged recall is limited to the labeled Top-6 windows, not whole-paper recall.",
        "",
        "| Config | Useful P@3 Survey / RP-2 / RP=1 | Partial P@3 Survey / RP-2 / RP=1 | Useful R@3 | Partial R@3 | U+P R@3 | MRR useful | Final-pool useful precision | Final partial share | Useful papers / q | ≥2 useful papers | All relevant U/P covered | All useful papers covered usefully | Irrelevant retained | Mean request estimate | Max | Mean BGE retrieval s/q | Mean rerank s/q |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in result["aggregate"].items():
        useful = " / ".join(f"{summary['precision_at_3_by_paper'][key]['useful']:.2f}" for key in paper_keys)
        partial = " / ".join(f"{summary['precision_at_3_by_paper'][key]['partial']:.2f}" for key in paper_keys)
        lines.append(
            f"| {name} | {useful} | {partial} | "
            f"{summary['recall_useful_at_3']:.2f} | {summary['recall_partial_at_3']:.2f} | "
            f"{summary['recall_useful_partial_at_3']:.2f} | {summary['mrr_first_useful']:.2f} | "
            f"{summary['pool_useful_precision']:.2f} | {summary['pool_partial_share']:.2f} | "
            f"{summary['mean_useful_papers']:.2f} | {summary['questions_with_2_useful_papers_pct']:.0f}% | "
            f"{summary['all_relevant_papers_covered_pct']:.0f}% | "
            f"{summary['all_useful_papers_covered_usefully_pct']:.0f}% | "
            f"{summary['pool_irrelevant_count']} | {summary['mean_request_estimate']:.0f} | "
            f"{summary['max_request_estimate']} | "
            f"{summary['mean_retrieval_seconds_per_question']:.3f} | "
            f"{summary['mean_rerank_seconds_per_question']:.3f} |"
        )
    lines.extend(["", "## Per-question selected evidence", ""])
    for qid, configs in result["per_question"].items():
        lines.append(f"**{qid}**")
        lines.append("")
        for name, metrics in configs.items():
            selected = ", ".join(
                f"{item['paper']}:{item['chunk']} ({item['label']})" for item in metrics["selected"]
            ) or "none"
            lines.append(f"- {name}: {selected}; estimate {metrics['estimate']} tokens")
        lines.append("")
    timing = result["timing"]
    lines.extend([
        "## Timing",
        "",
        f"- BGE indexing: {timing['index_seconds']:.2f}s.",
        f"- Mean base-query retrieval for A/C/D/E: {result['aggregate']['A']['mean_retrieval_seconds_per_question']:.3f}s/question; B with focused unions where available: {result['aggregate']['B']['mean_retrieval_seconds_per_question']:.3f}s/question.",
        f"- Cross-encoder cached load: {timing['cross_encoder_load_seconds']:.3f}s; mean rerank for D/E: {timing['mean_rerank_seconds_per_question']:.3f}s/question.",
        "- No generation-provider function is called.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--bge-model", default=os.environ.get("SCHOLARLENS_DEV_BGE_MODEL", "BAAI/bge-small-en-v1.5"))
    parser.add_argument("--cross-encoder-model", default=os.environ.get("SCHOLARLENS_DEV_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    parser.add_argument("--generation-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    result = run_evaluation(
        args.fixture, args.pdf_dir, args.bge_model,
        args.cross_encoder_model, args.generation_model, args.device,
    )
    report = _render_report(result)
    print(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Evaluation stopped: {error}", file=sys.stderr)
        raise SystemExit(2) from error

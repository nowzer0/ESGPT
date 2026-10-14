#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from esgpt.benchmark_evaluation import (
    annotate_prediction_usage,
    build_evaluators,
    choose_evaluator_name,
    discover_original_eval_code,
    evaluate_predictions_by_adapter,
    read_jsonl,
    write_json,
    write_jsonl,
    write_metric_breakdown_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score model predictions on ESGU-Bench.")
    parser.add_argument("--benchmark-file", required=True, help="Path to esgu_bench.jsonl.")
    parser.add_argument("--predictions-file", required=True, help="JSONL with sample_id and prediction fields.")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory for evaluation outputs.")
    return parser.parse_args()


def prepare_predictions(
    records: List[Dict[str, Any]],
    submitted_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    submitted_by_id: Dict[str, Dict[str, Any]] = {}
    for row in submitted_rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("Every prediction row must contain sample_id.")
        if sample_id in submitted_by_id:
            raise ValueError(f"Duplicate prediction sample_id: {sample_id}")
        submitted_by_id[sample_id] = row

    benchmark_ids = [str(record.get("sample_id") or "").strip() for record in records]
    if not all(benchmark_ids) or len(set(benchmark_ids)) != len(benchmark_ids):
        raise ValueError("Benchmark sample_id values must be present and unique.")

    missing = [sample_id for sample_id in benchmark_ids if sample_id not in submitted_by_id]
    unknown = sorted(set(submitted_by_id) - set(benchmark_ids))
    if missing or unknown:
        raise ValueError(
            f"Prediction coverage mismatch: missing={len(missing)}, unknown={len(unknown)}. "
            f"Examples: missing={missing[:3]}, unknown={unknown[:3]}"
        )

    predictions: List[Dict[str, Any]] = []
    for record, sample_id in zip(records, benchmark_ids):
        submitted = submitted_by_id[sample_id]
        raw_prediction = submitted.get("prediction", submitted.get("raw_prediction"))
        if raw_prediction is None:
            raise ValueError(f"Prediction {sample_id} must contain prediction or raw_prediction.")
        predictions.append(
            {
                "sample_id": sample_id,
                "record_id": record.get("record_id"),
                "task_name": record.get("task_name"),
                "category": record.get("category"),
                "source_folder": record.get("source_folder"),
                "source_dataset": record.get("source_dataset"),
                "evaluator_name": choose_evaluator_name(record),
                "raw_prediction": str(raw_prediction),
            }
        )
    return predictions


def main() -> int:
    args = parse_args()
    benchmark_file = Path(args.benchmark_file)
    predictions_file = Path(args.predictions_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(benchmark_file)
    submitted_rows = read_jsonl(predictions_file)
    predictions = prepare_predictions(records, submitted_rows)

    inventory = discover_original_eval_code(benchmark_file.parent, output_dir=output_dir)
    evaluators = build_evaluators(inventory)
    predictions = annotate_prediction_usage(predictions, evaluators)

    evaluator_results, dataset_specific_results = evaluate_predictions_by_adapter(
        records, predictions, evaluators
    )
    breakdowns = write_metric_breakdown_reports(
        records=records,
        predictions=predictions,
        evaluators=evaluators,
        output_dir=output_dir,
    )

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    summary = {
        "benchmark_file": str(benchmark_file),
        "predictions_file": str(predictions_file),
        "example_count": len(records),
        "evaluator_results": evaluator_results,
        "dataset_specific_results": dataset_specific_results,
        "task_count": len(breakdowns["metrics_by_task"]),
        "category_count": len(breakdowns["metrics_by_category"]),
    }
    write_json(output_dir / "evaluation_results.json", summary)

    print(f"Evaluated {len(records)} examples.")
    print(f"Results: {output_dir / 'evaluation_results.json'}")
    for name, metrics in evaluator_results.items():
        score = metrics.get("accuracy", metrics.get("micro_f1", metrics.get("token_f1")))
        print(f"{name}: {score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

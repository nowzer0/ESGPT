import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

from .data import get_prompt_and_answer, load_records_for_task, render_chat_messages, select_task_dirs
from .metrics import (
    canonical_text,
    list_of_dicts_to_tuple_set,
    list_to_text_set,
    qa_metrics,
    safe_json_loads,
    set_metrics,
    single_label_metrics,
)
from .serialization import namespace_to_json_dict
from .task_registry import core_task_group_display_name, infer_core_task_group


def _parse_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported dtype: {name}")
    return getattr(torch, name)


def infer_schema(task_name: str, gold_obj: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(gold_obj.keys())
    if keys == {"answer"}:
        return {"schema_type": "qa", "field": "answer"}
    if keys == {"labels"} and isinstance(gold_obj["labels"], list):
        return {"schema_type": "list_of_objects", "field": "labels"}
    if keys == {"action", "aspect"}:
        return {"schema_type": "action_with_aspect"}
    if keys == {"category", "aspects"}:
        return {"schema_type": "category_with_aspects"}
    if len(keys) == 1:
        field = next(iter(keys))
        value = gold_obj[field]
        if isinstance(value, str):
            return {"schema_type": "single_label", "field": field}
        if isinstance(value, list):
            return {"schema_type": "string_set", "field": field}
    raise ValueError(f"Unsupported schema for task={task_name}: keys={sorted(keys)}")


def primary_metric_name(schema: Dict[str, Any]) -> str:
    mapping = {
        "single_label": "accuracy",
        "string_set": "micro_f1",
        "list_of_objects": "micro_f1",
        "action_with_aspect": "action_accuracy",
        "category_with_aspects": "aspects_micro_f1",
        "qa": "token_f1",
    }
    return mapping[schema["schema_type"]]


def _looks_like_local_path(path_text: str) -> bool:
    expanded = Path(path_text).expanduser()
    return (
        expanded.exists()
        or expanded.is_absolute()
        or path_text.startswith(("./", "../", "~/", "~\\"))
        or path_text.startswith(("/", "\\"))
    )


def _quantization_config_name(quantization_config: Any) -> str:
    if isinstance(quantization_config, dict):
        return str(
            quantization_config.get("quant_method")
            or quantization_config.get("quantization_method")
            or quantization_config.get("load_in")
            or "dict"
        )
    return type(quantization_config).__name__


def _is_fp8_quantization_config(quantization_config: Any) -> bool:
    name = _quantization_config_name(quantization_config).lower()
    if "fp8" in name:
        return True
    if isinstance(quantization_config, dict):
        return any("fp8" in str(value).lower() for value in quantization_config.values())
    return "fp8" in repr(quantization_config).lower()


def _max_cuda_compute_capability() -> Optional[tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    capabilities = [torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())]
    return max(capabilities) if capabilities else None


def _pretrained_quantization_config(model_name_or_path: str, *, trust_remote_code: bool) -> Any:
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    except Exception as exc:
        print(
            f"WARNING: could not inspect model config for existing quantization_config: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
    return getattr(config, "quantization_config", None)


def load_inference_model_and_tokenizer(args: Any) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    dtype = _parse_dtype(args.torch_dtype)
    if dtype != "auto":
        model_kwargs["dtype"] = dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        existing_quantization_config = _pretrained_quantization_config(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        if existing_quantization_config is not None:
            print(
                "WARNING: model config already includes quantization_config "
                f"({_quantization_config_name(existing_quantization_config)}); "
                "ignoring --load-in-4bit to avoid quantization config conflict.",
                flush=True,
            )
            if _is_fp8_quantization_config(existing_quantization_config):
                capability = _max_cuda_compute_capability()
                if capability is None or capability < (8, 9):
                    capability_text = "no CUDA GPU" if capability is None else f"{capability[0]}.{capability[1]}"
                    raise ValueError(
                        "This checkpoint is already FP8-quantized, and Transformers requires GPUs with "
                        f"compute capability >= 8.9 for FineGrained FP8 models. Current maximum CUDA "
                        f"compute capability is {capability_text}. Use a non-FP8 checkpoint and load it "
                        "with --load-in-4bit, or run this FP8 checkpoint on a supported GPU such as RTX 4090/H100."
                    )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.adapter_path:
        from peft import PeftModel

        if _looks_like_local_path(str(args.adapter_path)):
            adapter_path = Path(args.adapter_path).expanduser()
            if not adapter_path.exists():
                raise ValueError(f"Adapter path does not exist: '{adapter_path}'")
            adapter_config_path = adapter_path / "adapter_config.json"
            if not adapter_config_path.is_file():
                raise ValueError(
                    f"Adapter path exists but '{adapter_config_path.name}' is missing at '{adapter_path}'. "
                    "Point --adapter-path to a saved PEFT adapter directory."
                )
            model = PeftModel.from_pretrained(model, str(adapter_path))
        else:
            model = PeftModel.from_pretrained(model, args.adapter_path)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        for attr_name in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, attr_name):
                setattr(model.generation_config, attr_name, None)
    model.eval()
    return model, tokenizer


def _left_pad_batch(sequences: Sequence[List[int]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = []
    attention_mask = []
    for sequence in sequences:
        pad_size = max_length - len(sequence)
        input_ids.append(([pad_token_id] * pad_size) + sequence)
        attention_mask.append(([0] * pad_size) + ([1] * len(sequence)))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_from_prompt_texts(
    model: Any,
    tokenizer: Any,
    prompt_texts: Sequence[str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    progress_desc: Optional[str] = None,
) -> List[str]:
    outputs: List[str] = []
    model_device = model.device
    if do_sample and seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    prompt_text_list = list(prompt_texts)
    batch_starts = range(0, len(prompt_text_list), batch_size)
    total_batches = (len(prompt_text_list) + batch_size - 1) // batch_size if prompt_text_list else 0
    if progress_desc and total_batches:
        batch_starts = tqdm(
            batch_starts,
            total=total_batches,
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
        )

    for start in batch_starts:
        prompt_batch = prompt_text_list[start : start + batch_size]
        prompt_token_ids: List[List[int]] = []
        for prompt_text in prompt_batch:
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_token_ids.append(prompt_ids[-max_input_length:])

        batch_tensors = _left_pad_batch(prompt_token_ids, tokenizer.pad_token_id)
        batch_tensors = {key: value.to(model_device) for key, value in batch_tensors.items()}

        with torch.no_grad():
            generation_kwargs: Dict[str, Any] = {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                if temperature is not None:
                    generation_kwargs["temperature"] = temperature
                if top_p is not None:
                    generation_kwargs["top_p"] = top_p
                if top_k is not None:
                    generation_kwargs["top_k"] = top_k
            generated = model.generate(
                **batch_tensors,
                **generation_kwargs,
            )

        prompt_width = batch_tensors["input_ids"].shape[1]
        for index in range(len(prompt_batch)):
            generated_ids = generated[index][prompt_width:]
            outputs.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return outputs


def generate_task_predictions(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    chat_template_enable_thinking: Optional[bool] = None,
    progress_desc: Optional[str] = None,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    prompt_texts: List[str] = []
    prompt_token_lengths: List[int] = []

    for record in records:
        prompt_messages, _ = get_prompt_and_answer(record["messages"])
        prompt_text = render_chat_messages(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            enable_thinking=chat_template_enable_thinking,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompt_texts.append(prompt_text)
        prompt_token_lengths.append(min(len(prompt_ids), max_input_length))

    generated_texts = generate_from_prompt_texts(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        progress_desc=progress_desc,
    )

    for record, prompt_text, prompt_length, generated_text in zip(
        records, prompt_texts, prompt_token_lengths, generated_texts
    ):
        predictions.append(
            {
                "id": record.get("id"),
                "detail_task": record["detail_task"],
                "task": record.get("task"),
                "core_task_group": record["core_task_group"],
                "prompt_text": prompt_text,
                "raw_prediction": generated_text,
                "prompt_tokens": prompt_length,
            }
        )
    return predictions


def score_task_records(
    records: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
    if len(records) != len(predictions):
        raise ValueError("Record/prediction length mismatch.")

    gold_objects: List[Dict[str, Any]] = []
    pred_objects: List[Optional[Dict[str, Any]]] = []
    invalid_json_count = 0
    for record, prediction in zip(records, predictions):
        gold_text = record["messages"][-1]["content"]
        gold_obj, gold_error = safe_json_loads(gold_text)
        if gold_obj is None:
            raise ValueError(f"Gold JSON could not be parsed for task={record.get('task')}: {gold_error}")
        gold_objects.append(gold_obj)

        pred_obj, pred_error = safe_json_loads(prediction["raw_prediction"])
        if pred_obj is None:
            invalid_json_count += 1
        pred_objects.append(pred_obj)
        prediction["parsed_prediction"] = pred_obj
        prediction["json_error"] = pred_error
        prediction["gold"] = gold_obj

    schema = infer_schema(records[0]["task"], gold_objects[0])
    schema_type = schema["schema_type"]
    metrics: Dict[str, float]

    if schema_type == "single_label":
        field = schema["field"]
        golds = [canonical_text(item[field]) for item in gold_objects]
        preds = [
            canonical_text(pred[field]) if isinstance(pred, dict) and field in pred else ""
            for pred in pred_objects
        ]
        metrics = single_label_metrics(golds, preds)
    elif schema_type == "string_set":
        field = schema["field"]
        gold_sets = [list_to_text_set(item[field]) for item in gold_objects]
        pred_sets = [
            list_to_text_set(pred.get(field, [])) if isinstance(pred, dict) else set()
            for pred in pred_objects
        ]
        metrics = set_metrics(gold_sets, pred_sets)
    elif schema_type == "list_of_objects":
        field = schema["field"]
        gold_sets = [list_of_dicts_to_tuple_set(item[field]) for item in gold_objects]
        pred_sets = [
            list_of_dicts_to_tuple_set(pred.get(field, [])) if isinstance(pred, dict) else set()
            for pred in pred_objects
        ]
        metrics = set_metrics(gold_sets, pred_sets)
    elif schema_type == "action_with_aspect":
        gold_actions = [canonical_text(item["action"]) for item in gold_objects]
        pred_actions = [
            canonical_text(pred["action"]) if isinstance(pred, dict) and "action" in pred else ""
            for pred in pred_objects
        ]
        action_metrics = single_label_metrics(gold_actions, pred_actions)
        aspect_accuracy = sum(
            1
            for gold, pred in zip(gold_objects, pred_objects)
            if isinstance(pred, dict) and canonical_text(pred.get("aspect", "")) == canonical_text(gold["aspect"])
        ) / len(gold_objects)
        joint_exact = sum(
            1
            for gold, pred in zip(gold_objects, pred_objects)
            if isinstance(pred, dict)
            and canonical_text(pred.get("action", "")) == canonical_text(gold["action"])
            and canonical_text(pred.get("aspect", "")) == canonical_text(gold["aspect"])
        ) / len(gold_objects)
        metrics = {
            "action_accuracy": action_metrics["accuracy"],
            "action_macro_f1": action_metrics["macro_f1"],
            "aspect_accuracy": aspect_accuracy,
            "joint_exact_match": joint_exact,
        }
    elif schema_type == "category_with_aspects":
        category_accuracy = sum(
            1
            for gold, pred in zip(gold_objects, pred_objects)
            if isinstance(pred, dict) and canonical_text(pred.get("category", "")) == canonical_text(gold["category"])
        ) / len(gold_objects)
        gold_sets = [list_to_text_set(item["aspects"]) for item in gold_objects]
        pred_sets = [
            list_to_text_set(pred.get("aspects", [])) if isinstance(pred, dict) else set()
            for pred in pred_objects
        ]
        aspect_metrics = set_metrics(gold_sets, pred_sets)
        joint_exact = sum(
            1
            for gold, pred, gold_set, pred_set in zip(gold_objects, pred_objects, gold_sets, pred_sets)
            if isinstance(pred, dict)
            and canonical_text(pred.get("category", "")) == canonical_text(gold["category"])
            and gold_set == pred_set
        ) / len(gold_objects)
        metrics = {
            "category_accuracy": category_accuracy,
            "aspects_exact_match": aspect_metrics["exact_match"],
            "aspects_micro_f1": aspect_metrics["micro_f1"],
            "joint_exact_match": joint_exact,
        }
    elif schema_type == "qa":
        gold_answers = [str(item["answer"]) for item in gold_objects]
        pred_answers = [
            str(pred["answer"]) if isinstance(pred, dict) and "answer" in pred else ""
            for pred in pred_objects
        ]
        metrics = qa_metrics(gold_answers, pred_answers)
    else:
        raise ValueError(f"Unsupported schema type: {schema_type}")

    metrics["invalid_json_rate"] = invalid_json_count / len(records)
    metrics["example_count"] = float(len(records))
    metrics["primary_score"] = metrics[primary_metric_name(schema)]
    annotated_predictions = list(predictions)
    return metrics, annotated_predictions, schema


def summarize_group_scores(task_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = {}
    for row in task_rows:
        grouped.setdefault(row["core_task_group"], []).append(row["primary_score"])
    return {
        group_name: {
            "display_name": core_task_group_display_name(group_name),
            "macro_primary_score": sum(scores) / len(scores),
            "task_count": float(len(scores)),
        }
        for group_name, scores in grouped.items()
    }


def evaluate_selected_tasks_with_model(model: Any, tokenizer: Any, args: Any) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    wall_start = time.perf_counter()

    task_dirs = select_task_dirs(
        dataset_root=dataset_root,
        detail_tasks=args.detail_tasks,
        core_task_groups=args.core_task_groups,
        exclude_detail_tasks=args.exclude_detail_tasks,
    )
    if not task_dirs:
        raise ValueError("No task directories matched the selection arguments.")

    task_rows: List[Dict[str, Any]] = []
    total_tasks = len(task_dirs)

    for task_index, task_dir in enumerate(task_dirs, start=1):
        records = load_records_for_task(
            task_dir=task_dir,
            split=args.split,
            max_samples=args.max_samples_per_task,
            seed=args.seed,
        )
        print(
            f"[{task_index}/{total_tasks}] Evaluating {task_dir.name} "
            f"on split={args.split} ({len(records)} examples)",
            flush=True,
        )
        predictions = generate_task_predictions(
            model=model,
            tokenizer=tokenizer,
            records=records,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            chat_template_enable_thinking=args.chat_template_enable_thinking,
            progress_desc=f"{task_index}/{total_tasks} {task_dir.name}",
        )
        metrics, annotated_predictions, schema = score_task_records(records, predictions)
        task_row = {
            "detail_task": task_dir.name,
            "task": records[0]["task"],
            "core_task_group": infer_core_task_group(task_dir.name, task_name=records[0]["task"]),
            "core_task_group_display": core_task_group_display_name(
                infer_core_task_group(task_dir.name, task_name=records[0]["task"])
            ),
            "schema_type": schema["schema_type"],
            **metrics,
        }
        task_rows.append(task_row)

        prediction_path = prediction_dir / f"{task_dir.name}_{args.split}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for item in annotated_predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"[{task_index}/{total_tasks}] Completed {task_dir.name}: "
            f"primary_score={task_row['primary_score']:.4f}, "
            f"invalid_json_rate={metrics['invalid_json_rate']:.4f}",
            flush=True,
        )

    group_summary = summarize_group_scores(task_rows)
    overall_macro_primary = (
        sum(row["primary_score"] for row in task_rows) / len(task_rows) if task_rows else 0.0
    )
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "dataset_root": str(dataset_root),
        "split": args.split,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": time.perf_counter() - wall_start,
        "task_count": len(task_rows),
        "overall_macro_primary_score": overall_macro_primary,
        "task_rows": task_rows,
        "group_summary": group_summary,
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "evaluation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(namespace_to_json_dict(args), handle, ensure_ascii=False, indent=2)
    return summary


def evaluate_selected_tasks(args: Any) -> Dict[str, Any]:
    model, tokenizer = load_inference_model_and_tokenizer(args)
    return evaluate_selected_tasks_with_model(model=model, tokenizer=tokenizer, args=args)

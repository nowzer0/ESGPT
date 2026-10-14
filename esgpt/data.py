import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .task_registry import core_task_group_display_name, infer_core_task_group, normalize_core_task_group_name

SPLIT_ALIASES = {
    "train": ("train",),
    "validation": ("validation", "val", "valid"),
    "val": ("val", "validation", "valid"),
    "valid": ("valid", "validation", "val"),
    "test": ("test",),
    "all": ("all",),
}


def canonical_split_name(split: str) -> str:
    split = split.strip().lower()
    if split in {"validation", "val", "valid"}:
        return "validation"
    return split


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def _has_split_messages_file(path: Path) -> bool:
    for split_aliases in SPLIT_ALIASES.values():
        for alias in split_aliases:
            if any(path.glob(f"{alias}_*_messages.jsonl")):
                return True
            if (path / f"{alias}_messages.jsonl").exists():
                return True
    return False


def _is_hidden_relative_path(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part.startswith(".") for part in relative_parts)


def discover_detail_task_dirs(dataset_root: Path) -> List[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    immediate_task_dirs = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and _has_split_messages_file(path)
    )
    if immediate_task_dirs:
        return immediate_task_dirs
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_dir()
        and not _is_hidden_relative_path(path, dataset_root)
        and _has_split_messages_file(path)
    )


def resolve_split_files(task_dir: Path, split: str, file_format: str = "messages") -> List[Path]:
    requested_split = canonical_split_name(split)
    aliases = SPLIT_ALIASES.get(requested_split, (requested_split,))
    candidates: List[Path] = []
    for alias in aliases:
        exact_path = task_dir / f"{alias}_{file_format}.jsonl"
        if exact_path.exists():
            candidates.append(exact_path)
        candidates.extend(sorted(task_dir.glob(f"{alias}_*_{file_format}.jsonl")))
    if requested_split == "all" and not candidates:
        candidates.extend(sorted(task_dir.glob(f"*_{file_format}.jsonl")))
    candidates = sorted(set(candidates))
    if not candidates:
        available = sorted(path.name for path in task_dir.glob(f"*_{file_format}.jsonl"))
        raise FileNotFoundError(
            f"No {file_format} file for split='{split}' in {task_dir}. Available: {available}"
        )
    return candidates


def resolve_split_file(task_dir: Path, split: str, file_format: str = "messages") -> Path:
    candidates = resolve_split_files(task_dir, split, file_format)
    return candidates[0]


def select_task_dirs(
    dataset_root: Path,
    detail_tasks: Optional[Sequence[str]] = None,
    core_task_groups: Optional[Sequence[str]] = None,
    exclude_detail_tasks: Optional[Sequence[str]] = None,
) -> List[Path]:
    task_dirs = discover_detail_task_dirs(dataset_root)
    if detail_tasks:
        allowed = set(detail_tasks)
        task_dirs = [path for path in task_dirs if path.name in allowed]
    if core_task_groups:
        allowed_groups = {normalize_core_task_group_name(group_name) for group_name in core_task_groups}
        task_dirs = [
            path
            for path in task_dirs
            if infer_core_task_group(path.name) in allowed_groups
        ]
    if exclude_detail_tasks:
        blocked = set(exclude_detail_tasks)
        task_dirs = [path for path in task_dirs if path.name not in blocked]
    return task_dirs


def augment_record(record: Dict[str, Any], task_dir: Path) -> Dict[str, Any]:
    enriched = dict(record)
    enriched["detail_task"] = task_dir.name
    enriched["core_task_group"] = infer_core_task_group(task_dir.name, task_name=record.get("task"))
    enriched["core_task_group_display"] = core_task_group_display_name(enriched["core_task_group"])
    enriched["task_dir"] = str(task_dir)
    return enriched


def load_records_for_task(
    task_dir: Path,
    split: str,
    file_format: str = "messages",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for split_file in resolve_split_files(task_dir, split, file_format):
        records.extend(augment_record(record, task_dir) for record in read_jsonl(split_file))
    if max_samples is not None and len(records) > max_samples:
        rng = random.Random(seed)
        records = rng.sample(records, max_samples)
    return records


def load_records(
    task_dirs: Sequence[Path],
    split: str,
    file_format: str = "messages",
    max_samples_per_task: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    all_records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for task_dir in task_dirs:
        task_records = load_records_for_task(
            task_dir=task_dir,
            split=split,
            file_format=file_format,
            max_samples=max_samples_per_task,
            seed=seed,
        )
        all_records.extend(task_records)
        counts[task_dir.name] = len(task_records)
    return all_records, counts


def get_prompt_and_answer(messages: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Expected a final assistant turn in the messages list.")
    prompt_messages = list(messages[:-1])
    answer_text = messages[-1]["content"]
    return prompt_messages, answer_text


def render_fallback_chat(messages: Sequence[Dict[str, str]], add_generation_prompt: bool) -> str:
    rendered: List[str] = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def render_chat_messages(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: Optional[bool] = None,
) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        template_kwargs: Dict[str, Any] = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
        except TypeError:
            if "enable_thinking" not in template_kwargs:
                raise
            return tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    return render_fallback_chat(messages, add_generation_prompt=add_generation_prompt)

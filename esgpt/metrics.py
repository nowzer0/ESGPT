import json
import re
import string
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def safe_json_loads(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(text), None
    except Exception:
        candidate = extract_first_json_object(text)
        if candidate is None:
            return None, "no_json_object_found"
        try:
            return json.loads(candidate), None
        except Exception as exc:
            return None, f"json_parse_error:{exc}"


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).strip().split())


def canonical_text(text: Any) -> str:
    return normalize_whitespace(str(text)).lower()


def normalize_answer_text(text: Any) -> str:
    value = str(text).lower()
    value = "".join(char for char in value if char not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return normalize_whitespace(value)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def single_label_metrics(golds: Sequence[str], preds: Sequence[str]) -> Dict[str, float]:
    total = len(golds)
    labels = sorted(set(golds) | set(preds))
    accuracy = sum(1 for gold, pred in zip(golds, preds) if gold == pred) / total if total else 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0
    counts = Counter(golds)
    for label in labels:
        tp = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(golds, preds) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred != label)
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        macro_f1 += f1
        weighted_f1 += f1 * counts.get(label, 0)
    if labels:
        macro_f1 /= len(labels)
    if total:
        weighted_f1 /= total
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def list_to_text_set(items: Iterable[Any]) -> Set[str]:
    return {canonical_text(item) for item in items if canonical_text(item)}


def list_of_dicts_to_tuple_set(items: Iterable[Dict[str, Any]]) -> Set[Tuple[Tuple[str, str], ...]]:
    output: Set[Tuple[Tuple[str, str], ...]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = tuple(sorted((str(key), canonical_text(value)) for key, value in item.items()))
        if normalized:
            output.add(normalized)
    return output


def set_metrics(gold_sets: Sequence[Set[Any]], pred_sets: Sequence[Set[Any]]) -> Dict[str, float]:
    total = len(gold_sets)
    exact = 0
    jaccard_sum = 0.0
    sample_f1_sum = 0.0
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    for gold_set, pred_set in zip(gold_sets, pred_sets):
        if gold_set == pred_set:
            exact += 1
        intersection = len(gold_set & pred_set)
        union = len(gold_set | pred_set)
        jaccard_sum += (intersection / union) if union else 1.0
        tp = intersection
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        sample_f1_sum += f1
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    micro_precision, micro_recall, micro_f1 = precision_recall_f1(micro_tp, micro_fp, micro_fn)
    return {
        "exact_match": exact / total if total else 0.0,
        "sample_f1": sample_f1_sum / total if total else 0.0,
        "mean_jaccard": jaccard_sum / total if total else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }


def token_f1(gold: str, pred: str) -> float:
    gold_tokens = normalize_answer_text(gold).split()
    pred_tokens = normalize_answer_text(pred).split()
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    overlap = Counter(gold_tokens) & Counter(pred_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(gold: str, pred: str) -> float:
    try:
        scorer = _rouge_scorer()
        return float(scorer.score(str(gold), str(pred))["rougeL"].fmeasure)
    except RuntimeError:
        gold_tokens = _metric_tokens(gold)
        pred_tokens = _metric_tokens(pred)
        if not gold_tokens and not pred_tokens:
            return 1.0
        if not gold_tokens or not pred_tokens:
            return 0.0
        lcs = lcs_length(gold_tokens, pred_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(gold_tokens)
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))


def _metric_tokens(text: Any) -> List[str]:
    return normalize_answer_text(text).split()


def _missing_metric_package_error(package: str, exc: Exception) -> RuntimeError:
    install_command = "pip install bert-score nltk rouge-score sacrebleu"
    return RuntimeError(
        f"Missing generation metric package '{package}'. Install metric dependencies with: "
        f"`{install_command}`"
    )


@lru_cache(maxsize=1)
def _rouge_scorer() -> Any:
    try:
        from rouge_score import rouge_scorer
    except Exception as exc:
        raise _missing_metric_package_error("rouge-score", exc) from exc
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


@lru_cache(maxsize=3)
def _bleu_metric(max_order: int) -> Any:
    try:
        from sacrebleu.metrics import BLEU
    except Exception as exc:
        raise _missing_metric_package_error("sacrebleu", exc) from exc
    return BLEU(max_ngram_order=max_order, effective_order=True, smooth_method="exp")


class _NoWordNet:
    def synsets(self, word: str) -> List[Any]:
        return []


def bleu_n(gold: str, pred: str, max_order: int) -> float:
    if not str(gold).strip() or not str(pred).strip():
        return 0.0
    return float(_bleu_metric(max_order).sentence_score(str(pred), [str(gold)]).score / 100.0)


def meteor_score(gold: str, pred: str) -> float:
    try:
        from nltk.translate.meteor_score import single_meteor_score
    except Exception as exc:
        raise _missing_metric_package_error("nltk", exc) from exc

    gold_tokens = _metric_tokens(gold)
    pred_tokens = _metric_tokens(pred)
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    try:
        from nltk.corpus import wordnet

        return float(single_meteor_score(gold_tokens, pred_tokens, wordnet=wordnet))
    except LookupError:
        # Keep evaluation runnable when NLTK corpora were not downloaded yet.
        return float(single_meteor_score(gold_tokens, pred_tokens, wordnet=_NoWordNet()))


def generation_overlap_metrics(gold: str, pred: str) -> Dict[str, float]:
    def safe_score(fn: Any) -> float:
        try:
            return float(fn())
        except RuntimeError:
            return 0.0

    return {
        "rouge_l_f1": rouge_l_f1(gold, pred),
        "meteor": safe_score(lambda: meteor_score(gold, pred)),
        "bleu_1": safe_score(lambda: bleu_n(gold, pred, 1)),
        "bleu_2": safe_score(lambda: bleu_n(gold, pred, 2)),
        "bleu_3": safe_score(lambda: bleu_n(gold, pred, 3)),
    }


def qa_metrics(golds: Sequence[str], preds: Sequence[str]) -> Dict[str, float]:
    total = len(golds)
    exact = 0
    token_f1_sum = 0.0
    rouge_sum = 0.0
    for gold, pred in zip(golds, preds):
        exact += int(normalize_answer_text(gold) == normalize_answer_text(pred))
        token_f1_sum += token_f1(gold, pred)
        rouge_sum += rouge_l_f1(gold, pred)
    return {
        "exact_match": exact / total if total else 0.0,
        "token_f1": token_f1_sum / total if total else 0.0,
        "rouge_l_f1": rouge_sum / total if total else 0.0,
    }

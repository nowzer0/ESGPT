import argparse
import json
from typing import Optional, Sequence

from .paths import DEFAULT_BENCHMARK_FULL_FILE, DEFAULT_DATASET_ROOT


def _add_model_loading_args(parser: argparse.ArgumentParser, *, require_model: bool = True) -> None:
    parser.add_argument("--model-name-or-path", required=require_model)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="chat_template_enable_thinking",
        action="store_true",
        help="Pass enable_thinking=True to the tokenizer chat template when supported.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="chat_template_enable_thinking",
        action="store_false",
        help="Pass enable_thinking=False to the tokenizer chat template when supported.",
    )
    parser.set_defaults(chat_template_enable_thinking=None)


def _add_generation_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--do-sample", action="store_true", help="Enable stochastic decoding.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic decoding.")


def _run_benchmark_subset_eval(args: argparse.Namespace) -> int:
    from .benchmark_evaluation import run_benchmark_subset_evaluation

    summary = run_benchmark_subset_evaluation(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _run_benchmark_subset_api_eval(args: argparse.Namespace) -> int:
    from .benchmark_evaluation import run_benchmark_subset_api_evaluation

    summary = run_benchmark_subset_api_evaluation(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m esgpt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_subset_eval_parser = subparsers.add_parser(
        "eval-benchmark-subset",
        help="Evaluate processed benchmark_full JSONL with dataset-specific adapters.",
    )
    _add_model_loading_args(benchmark_subset_eval_parser)
    benchmark_subset_eval_parser.add_argument(
        "--benchmark-file",
        default=str(DEFAULT_BENCHMARK_FULL_FILE),
        help="Path to esgu_bench.jsonl.",
    )
    benchmark_subset_eval_parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT.parent),
        help="Path to ESGPT_dataset for original evaluation code discovery.",
    )
    benchmark_subset_eval_parser.add_argument("--output-dir", required=True)
    benchmark_subset_eval_parser.add_argument("--adapter-path", default=None)
    benchmark_subset_eval_parser.add_argument("--batch-size", type=int, default=4)
    benchmark_subset_eval_parser.add_argument("--max-input-length", type=int, default=4096)
    benchmark_subset_eval_parser.add_argument("--max-new-tokens", type=int, default=256)
    benchmark_subset_eval_parser.add_argument(
        "--max-new-tokens-generation",
        type=int,
        default=2048,
        help="Longer generation limit for long-answer generation examples.",
    )
    benchmark_subset_eval_parser.add_argument("--max-samples", type=int, default=None)
    _add_generation_sampling_args(benchmark_subset_eval_parser)
    benchmark_subset_eval_parser.set_defaults(func=_run_benchmark_subset_eval)

    benchmark_subset_api_eval_parser = subparsers.add_parser(
        "eval-benchmark-subset-api",
        help="Evaluate processed benchmark JSONL through a DeepSeek/OpenAI-compatible chat completion API.",
    )
    benchmark_subset_api_eval_parser.add_argument(
        "--api-model",
        default="deepseek-v4-flash",
        help="API model name, e.g. deepseek-v4-flash, gpt-4.1, or gpt-5.1.",
    )
    benchmark_subset_api_eval_parser.add_argument(
        "--api-provider",
        choices=("deepseek", "openai", "openai_compatible"),
        default="deepseek",
        help="Provider-specific payload behavior. Use openai for api.openai.com models.",
    )
    benchmark_subset_api_eval_parser.add_argument(
        "--api-base-url",
        default="https://api.deepseek.com",
        help="OpenAI-compatible API base URL.",
    )
    benchmark_subset_api_eval_parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable that contains the API key.",
    )
    benchmark_subset_api_eval_parser.add_argument(
        "--api-key",
        default="",
        help="API key value. Prefer using --api-key-env to avoid logging secrets.",
    )
    benchmark_subset_api_eval_parser.add_argument(
        "--api-thinking",
        choices=("enabled", "disabled", "none"),
        default="disabled",
        help=(
            "DeepSeek thinking mode toggle. Disabled is recommended for strict benchmark labels. "
            "Use none for non-DeepSeek OpenAI-compatible servers such as vLLM."
        ),
    )
    benchmark_subset_api_eval_parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Reasoning effort. For OpenAI GPT-5.1, use none for low-latency non-reasoning mode.",
    )
    benchmark_subset_api_eval_parser.add_argument("--temperature", type=float, default=0.0)
    benchmark_subset_api_eval_parser.add_argument("--api-timeout", type=float, default=120.0)
    benchmark_subset_api_eval_parser.add_argument("--api-max-retries", type=int, default=5)
    benchmark_subset_api_eval_parser.add_argument("--api-retry-sleep", type=float, default=2.0)
    benchmark_subset_api_eval_parser.add_argument("--progress-every", type=int, default=25)
    benchmark_subset_api_eval_parser.add_argument(
        "--benchmark-file",
        default=str(DEFAULT_BENCHMARK_FULL_FILE),
        help="Path to esgu_bench.jsonl.",
    )
    benchmark_subset_api_eval_parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT.parent),
        help="Path to ESGPT_dataset for original evaluation code discovery.",
    )
    benchmark_subset_api_eval_parser.add_argument("--output-dir", required=True)
    benchmark_subset_api_eval_parser.add_argument("--max-new-tokens", type=int, default=256)
    benchmark_subset_api_eval_parser.add_argument(
        "--max-new-tokens-generation",
        type=int,
        default=2048,
        help="Longer generation limit for long-answer generation examples.",
    )
    benchmark_subset_api_eval_parser.add_argument("--max-samples", type=int, default=None)
    benchmark_subset_api_eval_parser.set_defaults(func=_run_benchmark_subset_api_eval)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

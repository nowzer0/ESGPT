from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "ESGPT_dataset" / "dataset_by_task"
DEFAULT_TRAIN_TEST_DATASET_ROOT = PROJECT_ROOT / "ESGPT_dataset" / "dataset_by_task_train_test"
DEFAULT_INSTRUCTION_CORPUS_ROOT = PROJECT_ROOT / "ESGPT_dataset" / "instruction_corpus_by_task"
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "ESGPT_dataset" / "benchmark_by_task"
DEFAULT_EXTERNAL_TEST_ROOT = PROJECT_ROOT / "ESGPT_dataset" / "dataset_by_task_external_test"
DEFAULT_EXTERNAL_INSTRUCTION_TEST_ROOT = PROJECT_ROOT / "ESGPT_dataset" / "dataset_by_task_external_instruction_test"
DEFAULT_CLIMATE_FINANCE_BENCH_ROOT = PROJECT_ROOT / "ESGPT_dataset" / "climate_finance_bench"
DEFAULT_CHATREPORT_ROOT = PROJECT_ROOT / "ESGPT_dataset" / "chatreport"
DEFAULT_PROCESSED_BENCHMARK_SUBSETS_ROOT = (
    PROJECT_ROOT / "ESGPT_dataset" / "processed_esg_data" / "benchmark_subsets"
)
DEFAULT_BENCHMARK_FULL_FILE = PROJECT_ROOT / "ESGPT_dataset" / "esgu_bench.jsonl"

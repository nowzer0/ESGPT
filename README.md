
# ESGPT

**Official repository for our EMNLP 2026 Main Conference paper:**

**ESGPT: A Data-Centric ESG Language Model with Unified Instruction and Benchmark Resources**

ESGPT provides unified resources for advancing and evaluating large language models in the Environmental, Social, and Governance (ESG) domain. The project introduces an ESG-specialized language model, an instruction-tuning dataset, and a comprehensive ESG understanding benchmark.

## Resources

The following resources are available on Hugging Face:

| Resource | Description | Hugging Face |
|---|---|---|
| **ESGPT** | An 8B ESG-specialized language model based on Qwen3-8B. | [Model](https://huggingface.co/nowzero/ESGPT) |
| **ESG-Instruct** | 30K instruction examples covering 35 instruction tasks and general instruction data. | [Dataset](https://huggingface.co/datasets/nowzero/ESG-Instruct) |
| **ESGU-Bench** | 9,514 evaluation examples across 21 tasks and six ESG understanding areas. | [Dataset](https://huggingface.co/datasets/nowzero/ESGU-Bench) |
| **ESG-EN** | Additional ESG dataset resource. | [Dataset](https://huggingface.co/datasets/nowzero/ESG-EN) |

## Model Performance

The figure below compares the overall accuracy of different language models on ESGU-Bench with respect to model size.

**ESGPT achieves 66.96% overall accuracy**, demonstrating competitive ESG understanding capabilities with substantially fewer parameters than many larger models.

![Model Performance](assets/model_performance.png)

## Quick Start

### Installation

Install the required libraries:

```bash
pip install -U transformers datasets accelerate torch
```

### Load ESGPT

ESGPT can be loaded directly from Hugging Face using the Transformers library.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "nowzero/ESGPT"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
```

### Load Datasets

The released datasets can be accessed using the Hugging Face Datasets library.

```python
from datasets import load_dataset

# ESG-Instruct
esg_instruct = load_dataset("nowzero/ESG-Instruct")

# ESGU-Bench
esgu_bench = load_dataset("nowzero/ESGU-Bench")

# ESG-EN
esg_en = load_dataset("nowzero/ESG-EN")
```

## ESGU-Bench Evaluation

This repository is dedicated to providing the official evaluation protocol for ESGU-Bench.

The evaluation toolkit will include:

- Task-specific evaluation prompts
- Model inference scripts
- Answer normalization and scoring procedures
- Evaluation configurations
- Result aggregation scripts

**Note:** This repository focuses on ESGU-Bench evaluation and reproducibility. ESGPT training code is not included.

Evaluation scripts and detailed instructions will be provided in a future update.

## License

The resources are distributed under their respective licenses, subject to applicable third-party terms.

- **ESGPT:** See the Hugging Face model repository.
- **ESG-Instruct:** See the Hugging Face dataset repository.
- **ESGU-Bench:** See the Hugging Face dataset repository.
- **ESG-EN:** See the Hugging Face dataset repository.
- **Evaluation Code:** Apache License 2.0 (planned).

## Citation

If you use ESGPT, ESG-Instruct, ESGU-Bench, or the associated evaluation resources, please cite our paper:

```bibtex
@inproceedings{lee2026esgpt,
  title     = {{ESGPT}: A Data-Centric {ESG} Language Model with Unified Instruction and Benchmark Resources},
  author    = {Lee, Jaeyoung and others},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```


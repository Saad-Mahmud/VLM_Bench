# VLM Bench (MCQ Dataset Pipeline)

This repository currently contains an MCQ-focused data preparation and prompting pipeline for three vision-language benchmarks:

- `mmlu_pro` (`MMMU/MMMU_Pro`, subset `vision`, split `test`)
- `science_qa` (`lmms-lab/ScienceQA`, subset `ScienceQA-IMG`, split `test`)
- `seed_bench` (`lmms-lab/SEED-Bench`, split `test`)

## Current Structure

Core scripts live in `dataset_util/mcq`:

- `dataset_util/mcq/*/stat.py`
- `dataset_util/mcq/*/download.py`
- `dataset_util/mcq/*/prompt_template.txt`
- `dataset_util/mcq/prompt_template.py`
- `dataset_util/mcq/mcq_dataloader.py`
- `tests/test_dataloader.py`

Generated data is written to `datasets/mcq/<dataset_name>/`:

- `samples.jsonl`
- `metadata.json`
- `images/`

## Environment

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install datasets pillow
```

## 1) Dataset Statistics

Each dataset has a `stat.py` that streams from Hugging Face and writes `report.json`.

```bash
python dataset_util/mcq/mmlu_pro/stat.py
python dataset_util/mcq/science_qa/stat.py
python dataset_util/mcq/seed_bench/stat.py
```

## 2) Sample Downloaders

Downloaders stream source data, apply dataset-specific filters, save images + normalized `samples.jsonl`, and write `metadata.json`.

```bash
python dataset_util/mcq/mmlu_pro/download.py -n 100 --seed 42
python dataset_util/mcq/science_qa/download.py -n 100 --seed 42
python dataset_util/mcq/seed_bench/download.py -n 100 --seed 42
```

Default output directories:

- `datasets/mcq/mmlu_pro`
- `datasets/mcq/science_qa`
- `datasets/mcq/seed_bench`

### Normalized Output Schema

All three downloaders save rows with:

- `id`
- `image`
- `question`
- `options`
- `answer`

Each downloader's `metadata.json` includes:

- output `columns`
- `source_columns`
- `column_creation` (how each output column is created)
- filter criteria and sampling details

## 3) Prompt Templates

Template files are stored in:

- `dataset_util/mcq/mmlu_pro/prompt_template.txt`
- `dataset_util/mcq/science_qa/prompt_template.txt`
- `dataset_util/mcq/seed_bench/prompt_template.txt`

Rules currently encoded:

- `mmlu_pro`: prompt assumes question/options are in image; answer must be `A` to `J`.
- `science_qa` and `seed_bench`: include textual `question` and options `A-D` from `options`.

## 4) Apply Templates to Raw Rows

Use `dataset_util/mcq/prompt_template.py` to render a prompt from either JSONL row or raw JSON:

```bash
python dataset_util/mcq/prompt_template.py \
  --dataset science_qa \
  --samples-file datasets/mcq/science_qa/samples.jsonl \
  --line-index 0
```

```bash
python dataset_util/mcq/prompt_template.py \
  --dataset seed_bench \
  --raw-query-json '{"question":"...","options":["a","b","c","d"]}'
```

## 5) Unified MCQ Dataloader

Use `dataset_util/mcq/mcq_dataloader.py` to iterate templated samples by dataset:

```bash
python dataset_util/mcq/mcq_dataloader.py --dataset mmlu_pro --limit 3
python dataset_util/mcq/mcq_dataloader.py --dataset science_qa --limit 3
python dataset_util/mcq/mcq_dataloader.py --dataset seed_bench --limit 3
```

This prints JSON lines containing:

- `id`
- `image`
- `answer`
- `prompt`

## 6) Quick End-to-End Sanity Check

`tests/test_dataloader.py` loads all three datasets with the shared dataloader and writes one rendered prompt per dataset to `example.txt`:

```bash
python tests/test_dataloader.py
```

## Notes

- `datasets/` is git-ignored by default in `.gitignore`.
- Download/stat scripts use streaming mode from Hugging Face datasets (no full dataset download step in code).

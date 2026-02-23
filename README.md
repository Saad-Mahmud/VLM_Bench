# VLM Bench

Benchmark open-source vision-language models (VLMs) on commonly used Visual Question Answering (VQA) benchmarks.

## Overview

This project is intended to provide a simple, reproducible framework for evaluating open-source VLMs on standard visual Q&A tasks.  
The goal is to compare model quality, speed, and robustness under a shared evaluation pipeline.

## Primary Goal

Evaluate open-source VLMs on commonly used visual Q&A benchmarks.

## Planned Scope

- Run multiple open-source VLMs with a consistent inference setup
- Evaluate on widely used VQA datasets/benchmarks
- Report core metrics in a comparable format
- Track experiment settings for reproducibility

## Candidate Benchmarks

- VQAv2
- GQA
- TextVQA
- VizWiz
- OK-VQA

## Evaluation Outputs

- Accuracy-style benchmark scores (dataset-specific)
- Per-model runtime/inference cost summaries
- Consolidated comparison tables

## Current Status

Project scaffold phase.  
Code, dataset loaders, model adapters, and evaluation scripts are being added.

## Roadmap

1. Add dataset ingestion and normalization
2. Add model adapter interface for open-source VLMs
3. Add unified evaluation runner
4. Add result aggregation and report generation
5. Add reproducibility tools (configs, seeds, logging)

## Contributing

Contributions are welcome.  
Please open an issue describing proposed changes before submitting a pull request.

## License

TBD

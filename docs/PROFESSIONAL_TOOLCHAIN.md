# Professional AI Engineering Toolchain

Daedalus follows the same lifecycle as a professional AI/ML system, but it starts with a
transparent, local NumPy engine. Production teams select additional tools according to the
problem; there is no single mandatory stack.

## Lifecycle comparison

| Engineering need | Common professional choices | Daedalus capability |
|---|---|---|
| Model development | PyTorch, JAX, TensorFlow/Keras, scikit-learn, Hugging Face | Inspectable tensor, layer, optimizer, and trainer primitives; optional frameworks are detected rather than bundled |
| Data identity and lineage | DVC, lakeFS, data catalogs, schema/quality systems | Validated numeric import, SHA-256 identity, deterministic split manifests, and leakage checks |
| Resolved configuration | `pyproject.toml`, `pylock.toml` or tool locks, Hydra | Standards-based project manifest, explicit experiment configuration, environment snapshot, and config hashes |
| Experiment tracking | MLflow, Weights & Biases, TensorBoard | SQLite run lifecycle, parameters, epoch events, metrics, checkpoint links, compact runtime/code provenance, and immutable run evidence |
| Model registry | MLflow Model Registry, artifact/object stores, Hugging Face Hub | Checksummed non-executable checkpoint generations and local evidence; external registries remain optional adapters |
| Evaluation | Task suites, slice/robustness checks, baselines, promotion gates | Exact held-out replay, confusion/per-class or residual metrics, baseline and direction-aware thresholds, and immutable reports |
| Compute and profiling | CUDA, ROCm, TPU, Accelerate, DeepSpeed, Ray, FSDP | CPU-safe teaching engine, memory/parameter calculators, profiler, and read-only capability detection |
| Serving | ONNX Runtime, FastAPI/BentoML, Triton, KServe, vLLM | Deployment and inference contracts prepared locally; production serving engines remain optional |
| Operations | OpenTelemetry, Prometheus/Grafana, drift and quality monitors | Structured local run events, automatic project logs, live diagnostics, and export-ready observability contracts |
| Quality and governance | Git, CI/CD, pytest, lint/type checks, dependency/secret scanning, SBOM/provenance, cards | Evidence gates, project smoke tests, Release Guard, model/data cards, backup, and recovery checks |

## Why Daedalus does not install every tool

PyTorch, TensorFlow, JAX, CUDA, DVC, MLflow, Docker, Kubernetes, and cloud platforms solve
different problems and can conflict in one environment. Daedalus owns the provider-neutral
workflow and evidence. Its Setup audit reports which compatible tools are installed and which
stage they help, without importing project code, installing packages, accessing accounts, or
sending private metadata to a service.

An optional tool being absent is not a failed project. A missing reproducibility artifact,
unvalidated dataset, unrecorded evaluation, or broken release gate is.

## Primary references

- [Google Cloud: MLOps continuous delivery and automation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/main/trainer)
- [DVC data and model versioning](https://dvc.org/doc/start/data-management/data-versioning)
- [Hydra configuration composition](https://hydra.cc/docs/intro/)
- [MLflow experiment tracking](https://mlflow.org/docs/latest/ml/tracking)
- [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards)
- [Python `pylock.toml` specification](https://packaging.python.org/en/latest/specifications/pylock-toml/)
- [OpenTelemetry signals](https://opentelemetry.io/docs/concepts/signals/)
- [GitHub Actions CI/CD](https://docs.github.com/en/actions/get-started/understand-github-actions)

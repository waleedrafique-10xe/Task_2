# GenAIQuant

A comprehensive Python package for model quantization that provides advanced tooling for efficient deployment and optimization of machine learning models. GenAIQuant supports multiple quantization algorithms and offers a flexible pipeline system for customizing model optimization workflows.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Quantization Pipeline](#quantization-pipeline)
- [Configuration Reference](#configuration-reference)
- [CLI Usage](#cli-usage)
- [Development](#development)
- [Known Issues](#known-issues)
- [Changelog](#changelog)

## Features

- **Multiple Quantization Algorithms**: Support for QuaRot, SpinQuant, GPTQ, EfficientQAT, and other state-of-the-art quantization methods
- **Flexible Pipeline System**: Chain multiple quantization techniques together for optimal results
- **Vision-Language Model (VLM) Support**: Quantize multimodal models like Qwen2.5-VL
- **Hugging Face Integration**: Seamless integration with Hugging Face model hub
- **Comprehensive Evaluation**: Built-in evaluation tools with multiple benchmark datasets (LMEval, VLMEvalKit, perplexity)
- **QDQ Hooks**: Support for quantization-dequantization hooks for both weights and activations
- **CLI & Python API**: Use via command line or programmatically

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (optional, but recommended for large models)
- Hugging Face account (for accessing gated models and datasets)
- Fast Hadamard Transform to be installed separately

### Hugging Face Authentication

Some models and datasets require special access permissions. You'll need to:

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)
2. **Request access** to gated models (e.g., Llama models require approval from Meta)
3. **Generate an access token**:
   - Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with "Read" permissions
4. **Set the token as an environment variable**:

```bash
# Linux/macOS
export HF_TOKEN=<your_token_here>

# Or add to your shell profile for persistence (~/.bashrc, ~/.zshrc, etc.)
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### Quick Install

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install GenAIQuant
pip install -e .
```

#### Installing Fast Hadamard Transform

```
pip install --no-build-isolation -e ".[fast]"
```

### Development Install

If you plan to modify GenAIQuant's source code or contribute to the project, use the development install. This includes additional tools for code quality, testing, and documentation:

```bash
# Install with development dependencies (linters, formatters, type checkers, etc.)
pip install -e .[dev]
```

**What's the difference?**

- **Standard install** (`pip install -e .`): Installs GenAIQuant in editable mode, meaning changes to the source code are immediately reflected without reinstalling. Ideal for using and experimenting with GenAIQuant.
- **Development install** (`pip install -e .[dev]`): Includes everything from standard install PLUS development tools like `black` (formatter), `ruff` (linter), `pyright` (type checker), and `pytest` (testing framework). Required if you want to run code quality checks or contribute code.

## Quick Start

### Basic Quantization

```bash
python3 -m GenAIQuant --model meta-llama/Llama-2-7b-hf --config configs/simple.yaml
```

### Available Configurations

Pre-configured quantization pipelines are available in the `configs` folder:

- `simple.yaml` - Basic quantization setup
- `qwen_vl_qat.json` - QAT for Qwen2.5-VL (language tower only)
- Additional configurations for various quantization strategies

### Example: Qwen2.5-VL QAT (Language Tower Only)

Quantize only the language tower of `Qwen/Qwen2.5-VL-3B-Instruct` while keeping the vision tower in full precision:

```bash
python -m GenAIQuant --config configs/qwen_vl_qat.json --model Qwen/Qwen2.5-VL-3B-Instruct
```

**Requirements:**

- Authenticated Hugging Face session (for dataset access)
- ~24GB GPU memory recommended
- Adjust `training_seqlen`, `train_size`, and `epochs` for different calibration budgets

## Supported Models

### Large Language Models (LLMs)

- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-3.2-1B`

### Vision-Language Models (VLMs)

- `Qwen/Qwen2.5-VL-3B-Instruct`

## Quantization Pipeline

GenAIQuant allows you to chain multiple quantization techniques in a customizable pipeline. The available optimization steps include:

### 1. Outlier Reduction

Handles activation outliers to improve quantization quality.

**Available Algorithms:**

- **QuaRot**: Rotation-based outlier reduction
- **SpinQuant**: Spin-based quantization with learned rotations

### 2. Bit Allocation

Flexible bit width allocation across model layers.

**Available Algorithms:**

- **ShortGPT**: Decoder block-level bit allocation

### 3. Post-Training Quantization (PTQ)

Efficient weight quantization without retraining.

**Available Algorithms:**

- **GPTQ**: Layerwise post-training quantization

### 4. Quantization Aware Training (QAT)

Fine-tuning with quantization in the loop.

**Available Algorithms:**

- **EfficientQAT**: Efficient QAT with BlockAP and E2E training
- **EfficientQAT with MBQ**: For vision-language models

### Example Pipeline Configuration

```yaml
quantization:
  pipeline:
    - outlier_reduction
    - bit_allocation
    - ptq
    - qat

  outlier_reduction:
    algorithm: SpinQuant
    online: true
    rotate_mode: hadamard

  bit_allocation:
    algorithm: ShortGpt
    target_avg_bitwidth: 3

  ptq:
    algorithm: Gptq
    wbits: 4
    groupsize: 64

  qat:
    algorithm: EfficientQat
    epochs: 2
    batch_size: 2
```

## Configuration Reference

### General Settings

| Parameter      | Type | Default      | Description                                         |
| -------------- | ---- | ------------ | --------------------------------------------------- |
| `device`     | str  | `"cuda:0"` | Device for quantization (`cpu`, `cuda:0`, etc.) |
| `output_dir` | Path | `"output"` | Directory for saving outputs and logs               |

### Evaluation Configuration

| Parameter                      | Type           | Default    | Description                                                                                 |
| ------------------------------ | -------------- | ---------- | ------------------------------------------------------------------------------------------- |
| `evaluation`                 | dict/null      | `null`   | Evaluation config (set to `null` to disable)                                              |
| `evaluation.tasks`           | dict[str, Any] | (required) | Mapping of `task_name -> task_config` (either `null` or a dict of task-specific kwargs) |
| `evaluation.device_map`      | str            | `"auto"` | Device placement strategy used when loading the model for evaluation                        |
| `evaluation.no_quantization` | bool           | `false`  | Disable both weight and activation quantization (evaluation-only)                           |
|                                |                |            |                                                                                             |
|                                |                |            |                                                                                             |
|                                |                |            |                                                                                             |

GenAIQuant routes tasks to the appropriate backend based on `task_name`:

- **LLM tasks (lm-eval)**: any task name known to `lm_eval` (e.g., `mmlu`, `arc_challenge`, `hellaswag`, `gsm8k`). You can pass kwargs like `num_fewshot`, `batch_size`, and other `lm_eval.simple_evaluate(...)` task args.
- **Perplexity (custom)**: `wikitext` supports `seq_len`, `batch_size`, and `num_samples`.
- **VLM tasks (VLMEvalKit)**: dataset names such as `MMBench_dev_en` or `TextVQA_VAL`. Each entry is a dict that typically includes `class` (e.g., `ImageMCQDataset`, `ImageVQADataset`) and `dataset` (usually the same string), plus optional dataset-specific args like `num_samples` (and video datasets may accept `fps`/`nframe`).

**Example: LLM evaluation (`configs/llama_3_2_1b/eqat.yaml`)**

```yaml
evaluation:
  no_quantization: false
  tasks:
    mmlu:
      num_fewshot: 5
    gsm8k:
      num_fewshot: 5
    arc_challenge: null
```

**Example: VLM evaluation (`configs/qwen2_5_vl_3b/spinquant_mbq.yaml`)**

```yaml
evaluation:
  no_quantization: false
  tasks:
    MMBench_dev_en:
      class: ImageMCQDataset
      dataset: MMBench_dev_en
    TextVQA_VAL:
      class: ImageVQADataset
      dataset: TextVQA_VAL
```

### Weight Quantization (`w_qconfig`)

Configure weight quantization separately for language and vision components:

| Parameter                   | Type     | Default | Description                             |
| --------------------------- | -------- | ------- | --------------------------------------- |
| `language.n_bits`         | int      | 4       | Weight bitwidth for language components |
| `language.group_size`     | int/null | 64      | Group size (`null` = per-channel)     |
| `language.is_two_power`   | bool     | false   | Restrict scales to powers of two        |
| `language.offset_enabled` | bool     | true    | Enable asymmetric quantization          |
| `vision.*`                | -        | varies  | Same parameters for vision tower        |

### Activation Quantization (`a_qconfig`)

Configure activation quantization with similar structure:

| Parameter                   | Type     | Default | Description                                |
| --------------------------- | -------- | ------- | ------------------------------------------ |
| `language.n_bits`         | int      | 8       | Activation bitwidth (4-8 typical)          |
| `language.group_size`     | int/null | 64      | Group size (`null` = per-tensor)         |
| `language.is_two_power`   | bool     | true    | Power-of-two scales for faster computation |
| `language.offset_enabled` | bool     | false   | Enable asymmetric quantization             |

### Algorithm-Specific Parameters

#### Outlier Reduction (SpinQuant/QuaRot)

| Parameter              | Type | Default         | Description                              |
| ---------------------- | ---- | --------------- | ---------------------------------------- |
| `algorithm`          | str  | `"SpinQuant"` | Algorithm choice                         |
| `online`             | bool | false           | Use runtime activation hooks             |
| `rotate_mode`        | str  | `"hadamard"`  | Rotation type (`hadamard`, `random`) |
| `w_bits`, `a_bits` | int  | 4               | Bitwidths for weights and activations    |
| `k_bits`, `v_bits` | int  | 4               | Bitwidths for attention keys and values  |
| `int8_down_proj`     | bool | false           | Use INT8 down-projection layers          |

#### Bit Allocation (ShortGPT)

| Parameter               | Type      | Default        | Description                        |
| ----------------------- | --------- | -------------- | ---------------------------------- |
| `algorithm`           | str       | `"ShortGpt"` | Algorithm choice                   |
| `target_avg_bitwidth` | float     | 3              | Target average bitwidth            |
| `bitwidth_options`    | list[int] | `[2,3,4]`    | Allowed bitwidth options           |
| `dataset_name`        | str       | `"wikitext"` | Calibration dataset                |
| `num_samples`         | int       | 128            | Number of calibration samples      |
| `seq_len`             | int       | 1024           | Sequence length per sample         |
| `n_prune_layers`      | int       | 6              | Layers to prune to lowest bitwidth |

#### Post-Training Quantization (GPTQ)

| Parameter       | Type | Default        | Description                       |
| --------------- | ---- | -------------- | --------------------------------- |
| `algorithm`   | str  | `"Gptq"`     | Algorithm choice                  |
| `datasetname` | str  | `"wikitext"` | Calibration dataset               |
| `numsamples`  | int  | 128            | Number of calibration samples     |
| `wbits`       | int  | 4              | Weight bitwidth                   |
| `sym`         | bool | false          | Use symmetric quantization        |
| `groupsize`   | int  | -1             | Group size (`-1` = per-channel) |

#### Quantization Aware Training (EfficientQAT)

| Parameter                | Type     | Default                     | Description                                                                                                                                                                                                              |
| ------------------------ | -------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `algorithm`            | str      | `"EfficientQat"`          | Algorithm choice                                                                                                                                                                                                         |
| `cache_dir`            | str      | `"./cache/EfficientQAT/"` | Cache directory                                                                                                                                                                                                          |
| `save_quant_dir`       | bool     | true                        | Save quantized model                                                                                                                                                                                                     |
| `calib_dataset`        | str      | `"wikitext"`              | Calibration dataset name/path. For text-only QAT, this is resolved via `DatasetProvider`. For multimodal BlockAP (`multimodal: true`), this is passed to `get_vl_dataset`. Options: `wikitext`, `redpajama` |
| `training_seqlen`      | int      | 2048                        | Training sequence length                                                                                                                                                                                                 |
| `train_size`           | int      | 4096                        | Number of training samples                                                                                                                                                                                               |
| `train_size_in_memory` | int/null | `null`                    | Optional RAM cap for calibration samples; enabling it activates a streaming ring-buffer loader that adds overhead, so keep it `null` unless the full dataset cannot fit in memory                                     |
| `val_size`             | int      | 64                          | Number of validation samples                                                                                                                                                                                             |
| `multimodal`           | bool     | false                       | Enable for VLMs; switches calibration over to `get_vl_dataset` and requires a processor                                                                                                                                |
| `batch_size`           | int      | 2                           | Training batch size                                                                                                                                                                                                      |
| `epochs`               | int      | 2                           | Number of training epochs                                                                                                                                                                                                |
| `wbits`                | int      | 4                           | Weight bitwidth                                                                                                                                                                                                          |
| `group_size`           | int      | 128                         | Quantization group size                                                                                                                                                                                                  |
| `quant_lr`             | float    | 1e-4                        | Learning rate for quantization params                                                                                                                                                                                    |
| `weight_lr`            | float    | 1e-5                        | Learning rate for model weights                                                                                                                                                                                          |
| `min_lr_factor`        | int      | 20                          | Min LR scaling factor                                                                                                                                                                                                    |
| `clip_grad`            | float    | 0.5                         | Gradient clipping threshold                                                                                                                                                                                              |
| `wd`                   | int      | 0                           | Weight decay                                                                                                                                                                                                             |
| `early_stop`           | int      | 0                           | Early stopping patience (`0` = disabled)                                                                                                                                                                               |
| `off_load_to_disk`     | bool     | false                       | Offload data to disk                                                                                                                                                                                                     |
| `e2e`                  | dict     | `{}`                      | E2E configuration (`qat.e2e`)                                                                                                                                                                                         |
| `mbq`                  | dict     | `{}`                      | MBQ configuration(`qat.mbq`)                                                                                                                                                                                           |

**When to use `train_size_in_memory`:** The default `null` value keeps the entire calibration set in RAM so blocks can reuse cached activations with minimal compute. Setting a positive value that is smaller than `train_size` switches EfficientQAT to the ring-buffer path, which streams smaller chunks through the model, recomputes activations each time, and therefore costs noticeably more wall-clock time. Only enable it when memory pressure forces you to; otherwise leave it unset for the best runtime/accuracy trade-off.

**E2E Configuration (`qat.e2e`)**

| Parameter                       | Type  | Default    | Description                                                      |
| ------------------------------- | ----- | ---------- | ---------------------------------------------------------------- |
| `dataset`                     | str   | `null`   | Dataset key for the E2E loader Option:`alpaca`                 |
| `pt_context_len`              | int   | `null`   | Context length used for pretraining-style datasets               |
| `gradient_checkpointing`      | bool  | `false`  | Enable PyTorch gradient checkpointing for memory-bound runs      |
| `learning_rate`               | float | `null`   | Learning rate for the E2E optimizer                              |
| `predict_with_generate`       | bool  | `false`  | Whether to call `generate` for validation                      |
| `eval_dataset_size`           | int   | `null`   | Number of evaluation samples                                     |
| `max_train_samples`           | int   | `null`   | Cap on training samples                                          |
| `max_eval_samples`            | int   | `null`   | Cap on eval samples                                              |
| `train_on_source`             | bool  | `false`  | If true, includes the source portion of SFT pairs in the loss    |
| `conv_temp`                   | str   | `null`   | Conversation template (e.g.,`llama-2`) for chat-style datasets |
| `wbits`                       | int   | `null`   | Optional override of the quantization bitwidth used in E2E       |
| `allow_multimodal`            | bool  | `false`  | Must be true to launch E2E for multimodal models                 |
| `per_device_train_batch_size` | int   | 1          | Per-device batch size for E2E training                           |
| `gradient_accumulation_steps` | int   | 16         | Gradient accumulation steps for E2E training                     |
| `max_grad_norm`               | float | 0.3        | Gradient clipping threshold for E2E training                     |
| `lr_scheduler_type`           | str   | `cosine` | Scheduler type passed into HF trainer                            |
| `warmup_ratio`                | float | 0.03       | Warmup ratio for the scheduler                                   |
| `logging_steps`               | int   | 10         | Logging interval for the trainer                                 |
| `seed`                        | int   | 0          | Seed used by the E2E trainer/data pipeline                       |

#### Quantization Aware Training with MBQ (EfficientQAT + MBQ)

**MBQ Configuration (`qat.mbq`)**

| Parameter                     | Type                                        | Default                       | Description                                                                                                                             |
| ----------------------------- | ------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`                   | bool                                        | `false`                     | Turns MBQ on (runs only for multimodal models and requires a working processor)                                                         |
| `dataset_name`              | str                                         | `lmms-lab/COCO-Caption2017` | Vision-language calibration dataset understood by `get_vl_dataset`. Options: `lmms-lab/COCO-Caption2017`, `AIMClab-RUC/COCO-CN` |
| `batch_size`                | int                                         | 2                             | Batch size for the KD loop                                                                                                              |
| `epochs`                    | int                                         | 2                             | Number of passes over the MBQ calibration set                                                                                           |
| `wbits`                     | int                                         | 4                             | Bitwidth for vision QuantLinear wrappers                                                                                                |
| `group_size`                | int                                         | 128                           | Group size for vision QuantLinear wrappers                                                                                              |
| `real_quant`                | bool                                        | `false`                     | Reserved for future use (not currently consumed by the MBQ loop)                                                                        |
| `teacher_device`            | str/null                                    | `null`                      | Device that holds the teacher copy (defaults to `quantization.device`)                                                                |
| `early_stop`                | int                                         | 0                             | Number of evaluation windows without improvement before stopping (0 disables)                                                           |
| `off_load_to_disk`          | bool                                        | `false`                     | Reserved for future use (not currently consumed by the MBQ loop)                                                                        |
| `mode`                      | `Literal["block","kd_block","global_kd"]` | `"block"`                   | Training mode selector                                                                                                                  |
| `kd_enabled`                | bool                                        | `true`                      | MBQ requires KD so this must remain true                                                                                                |
| `tau`                       | float                                       | 0.7                           | Temperature for the KD softmax                                                                                                          |
| `kd_weight`                 | float                                       | 1.0                           | Overall KD scaling factor                                                                                                               |
| `vision_weight`             | float                                       | 0.1                           | Relative weight assigned to image tokens during KD                                                                                      |
| `language_weight`           | float                                       | 1.0                           | Relative weight assigned to language tokens                                                                                             |
| `reweight`                  | bool                                        | `true`                      | Balance per-modality losses by token count                                                                                              |
| `train_size`                | int/null                                    | `null`                      | Override for MBQ train samples (falls back to `qat.train_size`)                                                                       |
| `val_size`                  | int/null                                    | `null`                      | Override for MBQ val samples (falls back to `qat.val_size`)                                                                           |
| `vision_token_ids`          | list[int]                                   | `[]`                        | Explicit image-token IDs; inferred from the processor when empty                                                                        |
| `vision_bitwidth_overrides` | dict[str, int]                              | `{}`                        | Optional per-module bitwidth overrides inside the vision tower                                                                          |

MBQ writes `mbq_best_scales.pt` and `mbq_final_scales.pt` into `output_dir` so you can reuse the learned quantizer parameters.

**Note:** MBQ runs only for multimodal models (`model.is_multimodal`) when `qat.mbq.enabled: true` and a processor can be loaded.

## CLI Usage

### Command Structure

```bash
python -m GenAIQuant --model <MODEL_NAME> --config <CONFIG_PATH> [OPTIONS]
```

### Required Arguments

- `--model MODEL`: Path to model or Hugging Face model name
- `--config CONFIG`: Path to configuration file (`.yaml` or `.json`)

### Optional Arguments

- `--skip-evaluation`: Skip QDQ and evaluation flow (default: `False`)
- `--skip-quantization`: Skip optimization pipeline, run evaluation only (default: `False`)
- `--evaluate-fp`: Evaluate model without quantization for baseline results (default: `False`)

### Example Commands

```bash
# Basic quantization with evaluation
python -m GenAIQuant --model meta-llama/Llama-2-7b-hf --config configs/simple.yaml

# Get baseline results (no quantization)
python -m GenAIQuant --model meta-llama/Llama-2-7b-hf --config configs/simple.yaml --evaluate-fp

# Quantize without evaluation
python -m GenAIQuant --model meta-llama/Llama-2-7b-hf --config configs/simple.yaml --skip-evaluation

# Evaluate pre-quantized model
python -m GenAIQuant --model path/to/quantized/model --config configs/simple.yaml --skip-quantization
```

## Development

### Code Quality Tools

```bash
# Format code with Ruff
ruff format .

# Type checking with Pyright
pyright

# Lint and auto-fix with Ruff
ruff check . --fix
```

### Building and Distribution

```bash
# Build wheel and source distribution
python -m build

# Output files
ls dist/
# GenAIQuant-0.1.0-py3-none-any.whl
# GenAIQuant-0.1.0.tar.gz
```

### Project Structure

```
GenAIQuant/
├── configs/              # Pre-configured quantization pipelines
├── GenAIQuant/           # Main package source
│   ├── algorithms/      # Quantization algorithm implementations
│   ├── evaluation/      # Evaluation tools and metrics
│   └── utils/          # Utility functions
├── tests/              # Test suite
└── docs/               # Documentation
```

## Known Issues

1. **[OWL-112](https://kinara.atlassian.net/browse/OWL-112)**: EQAT takes excessive time (~127h) when executed in pipeline with QuaRot online mode
2. **[OWL-116](https://kinara.atlassian.net/browse/OWL-116)**: GPTQ fails with QuaRot + ShortGPT for Llama-2 during Cholesky decomposition
3. **[OWL-135](https://kinara.atlassian.net/browse/OWL-135)**: R4 rotation currently not implemented in SpinQuant

## Changelog

### Version 0.4.1 (Current)

**New Features:**

- Adds ability to configure `max_steps` and `per_device_train_batch_size` for `SpinQuant` training
- Adds ability to load YAML format config files (JSON format still supported)
- Updates all existing configs to use YAML format
- Includes an `example.yaml` file with explanations on each configuration parameter

**Bug Fixes**:

- Fixes a bug in EfficientQAT implementation [OWL-189](https://kinara.atlassian.net/browse/OWL-189) causing NaN values during inference / evaluation

### Version 0.4.0

**New Features:**

- QDQ hooks for weight and activation quantization
- Support for Qwen2.5-VL and Llama model types
- Unified evaluation API integrating LMEvalKit, VLMEvalKit, and custom perplexity metrics
- Comprehensive QDQ parameter configuration (bitwidth, group size, power-of-two, asymmetric/symmetric)
- MBQ algorithm integration with EfficientQAT

### Version 0.3.1

**Features:**

- VLM support (Qwen2.5-VL) for SpinQuant and EfficientQAT
- Learned R1/R2 rotation matrices for VLM language sections
- Random Hadamard rotations for vision tower
- BlockAP and E2E training for VLM language sections
- VLM calibration data loading from HuggingFace datasets

**Bug Fixes:**

- Fixed VLMEvalKit issue preventing gradient descent on rotation matrices
- Resolved `pyproject.toml` issue with `pip install -e .`

### Version 0.3.0

**Features:**

- VLM support (Qwen2.5-VL) for QuaRot, GPTQ, and ShortGPT
- VLMEvalKit integration for VLM evaluation
- VLM calibration data loading capabilities

### Version 0.2.1

**Features:**

- SpinQuant (outlier reduction) support
- EfficientQAT (QAT) support

### Version 0.2.0

**Features:**

- Full pipeline support for QuaRot, GPTQ, and ShortGPT
- Model architecture-specific approach (removed `torch.export` dependency)
- Multi-GPU support

### Version 0.1.0

**Initial Release:**

- Core quantization framework
- Independent GPTQ and QuaRot support
- CLI interface and Python API
- LMEval integration
- Flexible configuration system

---

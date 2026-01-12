# GenAIQuant SDK – README

GenAIQuant provides Python-based quantization flows including EQAT, MBQ, SpinQuant, and Hadamard-enhanced quantization.

This SDK requires a CUDA-enabled environment.

---

## 1. System Requirements

### Python

- **Python ≥ 3.11** is required.

You can

```bash
python3 --version
```

### CUDA

- **CUDA > 12** is required for quantization.

### VRAM

- **VRAM ≥ 80GB** is recommended for smooth quantization of large models
- Recommended GPU is A100

### Disk Space

- **Available Disk Space ≥ 200GB** is required.

---

## 2. Installation Instructions

### Step 1: Create and Activate Python Virtual Environment

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install the GenAIQuant SDK Wheel

```bash
# add to your shell profile for persistence (~/.bashrc, ~/.zshrc, etc.)
pip3 install <SDK_ROOT_PATH>/GenAIQuant/GenAIQuant-0.4.1-py3-none-any.whl
```

---

## 3. Configuration File

configuration YAML is located at:

```
<SDK_ROOT_PATH>/GenAIQuant/configs
```

---

## 4. HuggingFace Token Setup

Some HuggingFace models require authentication.

### Linux/macOS

```bash
export HF_TOKEN="your_token_here"
```

Persist the token:

```bash
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. Install Fast Hadamard Transform (Required for CUDA Execution)

Before running quantization on CUDA devices , install the Hadamard transform package in the created Virtual Environment:

```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform
cd fast-hadamard-transform
pip install --no-build-isolation .
```

---

## 6. Running GenAIQuant

### General Command

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

### Example Usage

```bash
# Basic quantization with evaluation
python -m GenAIQuant --model <MODEL_NAME> --config <CONFIG_PATH>

# Get baseline results (no quantization)
python -m GenAIQuant --model <MODEL_NAME> --config <CONFIG_PATH> --evaluate-fp

# Quantize without evaluation
python -m GenAIQuant --model <MODEL_NAME> --config <CONFIG_PATH> --skip-evaluation

# Evaluate pre-quantized model
python -m GenAIQuant --model <MODEL_NAME> --config <CONFIG_PATH> --skip-quantization
```

# Quantizing Models

## Qwen/Qwen2.5-VL-3B-Instruct

### Quantization Flow 3: SpinQuant + EQAT w/ MBQ

```bash
python3 -m GenAIQuant --config <SDK_ROOT_PATH>/GenAIQuant/configs/qwen2_5_vl_3b/spinquant_mbq.yaml --model Qwen/Qwen2.5-VL-3B-Instruct
```

**If quantization is already complete and you have weights stored in `./output/spinquant_mbq/Qwen_Qwen2.5-VL-3B-Instruct/`, skip quantization and run evaluation only:**

```bash
python3 -m GenAIQuant --config <SDK_ROOT_PATH>/GenAIQuant/configs/qwen2_5_vl_3b_3b/spinquant_mbq.yaml --model Qwen/Qwen2.5-VL-3B-Instruct --skip-quantization
```

**Quantization Time:** TBD
**Evaluation Time:** TBD
**VRAM Requirements:** TBD

Output will be stored in `./output/spinquant_mbq/Qwen_Qwen2.5-VL-3B-Instruct/`

The output folder will contain the following subfolders

```
output/spinquant_mbq/Qwen_Qwen2.5-VL-3B-Instruct/
├── dequant/              # Containes dequantized model weights in safetensor format
                            along with model configs and processor files
├── dequant_eval
|   ├── vlm_eval.json     # Evaluation results on VLM tasks
└── rotation_matrices.pt  # Rotation matrices from SpinQuant
```

### Expected Scores

| Quantization Flow           | MMBench_EN | TextVQA |
| --------------------------- | ---------- | ------- |
| **Float Model (Baseline)**  | 79.0       | 77.608  |
| **SpinQuant + EQAT w/ MBQ** | 73.85      | 75.324  |

_Note: Higher scores are better for MMBench_EN and TextVQA._

This runs the full quantization workflow according to the provided configuration.

## meta-llama/Llama-3.2-3B-Instruct

### Quantization Flow : EQAT

```bash
python3 -m GenAIQuant --config <SDK_ROOT_PATH>/GenAIQuant/configs/llama_3_2_3b/eqat.yaml --model meta-llama/Llama-3.2-3B-Instruct
```

**If quantization is already complete and you have weights stored in `./output/eqat/meta-llama_Llama-3.2-3B-Instruct/`, skip quantization and run evaluation only:**

```bash
python3 -m GenAIQuant --config <SDK_ROOT_PATH>/GenAIQuant/configs/llama_3_2_3b/eqat.yaml --model meta-llama/Llama-3.2-3B-Instruct --skip-quantization
```

**Quantization Time:** TBD  
**Evaluation Time:** TBD  
**VRAM Requirements:** TBD

Output will be stored in `./output/eqat/meta-llama_Llama-3.2-3B-Instruct/`

The output folder will contain the following subfolders

```
output/eqat/meta-llama_Llama-3.2-3B-Instruct/
├── dequant/              # Contains dequantized model weights in safetensor format
                            along with model configs and processor files
└── dequant_eval
    ├── lm_eval.json      # Evaluation results on LM tasks
```

### Expected Scores

| Quantization Flow          | MMLU  | Arc Challenge | Arc Easy | boolq | hellaswag | openbookqa | piqa  | WinoGrande | gsm8k |
| -------------------------- | ----- | ------------- | -------- | ----- | --------- | ---------- | ----- | ---------- | ----- |
| **Float Model (Baseline)** | 59.58 | 43.34         | 73.91    | 78.47 | 52.21     | 27.6       | 76.01 | 67.25      | 64.75 |
| **EQAT**                   | 57.98 | 43.43         | 73.99    | 76.51 | 51.64     | 26.4       | 74.37 | 66.85      | 61.78 |

_Note: Higher scores are better for all metrics._

## meta-llama/Llama-3.1-8B-Instruct

### Quantization Flow : EQAT

```bash
python3 -m GenAIQuant --config <SDK_ROOT_PATH>/GenAIQuant/configs/llama_3_1_8b/eqat.yaml --model meta-llama/Llama-3.1-8B-Instruct
```

**If quantization is already complete and you have weights stored in `./output/eqat/meta-llama_Llama-3.1-8B-Instruct/`, skip quantization and run evaluation only:**

```bash
python3 -m GenAIQuant --config <SDK_ROOT_PATH>/GenAIQuant/configs/llama_3_1_8b/eqat.yaml --model meta-llama/Llama-3.1-8B-Instruct --skip-quantization
```

**Quantization Time:** TBD  
**Evaluation Time:** TBD  
**VRAM Requirements:** TBD

Output will be stored in `./output/eqat/meta-llama_Llama-3.1-8B-Instruct/`

The output folder will contain the following subfolders

```
output/eqat/meta-llama_Llama-3.1-8B-Instruct/
├── dequant/              # Contains dequantized model weights in safetensor format
                            along with model configs and processor files
└── dequant_eval
    ├── lm_eval.json      # Evaluation results on LM tasks
```

### Expected Scores

| Quantization Flow          | MMLU  | Arc Challenge | Arc Easy | boolq | hellaswag | openbookqa | piqa  | WinoGrande | gsm8k |
| -------------------------- | ----- | ------------- | -------- | ----- | --------- | ---------- | ----- | ---------- | ----- |
| **Float Model (Baseline)** | 68.1  | 51.9          | 81.7     | 84.1  | 59.1      | 33.4       | 80.0  | 73.9       | 75.7  |
| **EQAT**                   | 67.11 | 51.11         | 81.61    | 84.01 | 58.17     | 32.0       | 79.71 | 73.4       | 70.96 |

_Note: Higher scores are better for all metrics._

---

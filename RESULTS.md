## Quarot + GPTQ Benchmarks

| Model                                      | PPL    |
| ------------------------------------------ | ------ |
| Llama-2-7b                                 | 5.472  |
| Llama-2-7b (QuaRot) Offline                | 5.472  |
| Llama-2-7b (QuaRot) Online                 | 5.472  |
| Llama-2-7b (QuaRot+GPTQ) Offline Version   | 5.685  |
| Llama-2-7b (QuaRot+GPTQ) Online Version    | —      |
| Llama-3.2-1b                               | 9.748  |
| Llama-3.2-1b (QuaRot) Offline              | 9.748  |
| Llama-3.2-1b (QuaRot) Online               | 9.748  |
| Llama-3.2-1b (QuaRot+GPTQ) Offline Version | 11.032 |
| Llama-3.2-1b (QuaRot+GPTQ) Online Version  | 10.644 |

## ShortGpt + GPTQ Benchmarks

- `target_avg_bits` parameter determines the intial bit width allocation for all layers
- `n_prune_layers` parameter determines the number of layers to reduce to lowest bits ($2$)

| model                    | target_avg_bits | n_prune_layers | Completion Time (mm:ss) | perplexity |
| ------------------------ | --------------- | -------------- | ----------------------- | ---------- |
| meta-llama/Llama-2-7b-hf | 3               | 6              | 10:56                   | 11.373     |
| meta-llama/Llama-2-7b-hf | 4               | 6              | 11:03                   | 7.916      |
| meta-llama/Llama-2-7b-hf | 4               | 8              | 10:47                   | 10.462     |
| meta-llama/Llama-2-7b-hf | 4               | 4              | 10:52                   | 7.489      |
| meta-llama/Llama-2-7b-hf | 4               | 2              | 10:39                   | 6.429      |
| meta-llama/Llama-3.2-1B  | 3               | 6              | 3:03                    | 180.704    |
| meta-llama/Llama-3.2-1B  | 4               | 6              | 3:09                    | 118.615    |
| meta-llama/Llama-3.2-1B  | 4               | 8              | 3:03                    | 309.446    |
| meta-llama/Llama-3.2-1B  | 4               | 4              | 3:11                    | 43.108     |
| meta-llama/Llama-3.2-1B  | 4               | 2              | 3:12                    | 19.415     |

## Quarot + ShortGpt + GPTQ Benchmark

Benchmarked with the following configurations:

| Parameter                   | Value                                    |
| --------------------------- | ---------------------------------------- |
| Device                      | cuda:0                                   |
| Output Dir                  | benchmarks/dump_model                    |
| Evaluation Task             | WikiText                                 |
| Batch Size                  | 1                                        |
| Seq Length                  | 2048                                     |
| # Samples                   | 96                                       |
| Quantization Pipeline       | Outlier Reduction → Bit Allocation → PTQ |
| Outlier Reduction Algorithm | QuaRot                                   |
| PTQ Algorithm               | GPTQ                                     |
| PTQ Dataset                 | WikiText                                 |
| PTQ # Samples               | 64                                       |
| Bits (wbits)                | 4                                        |
| Symmetric                   | True                                     |
| Groupsize                   | 64                                       |
| Bit Allocation Algorithm    | ShortGPT                                 |
| Target Avg Bitwidth         | 4                                        |
| Pruned Layers               | 8                                        |

The results are as follows:

| Model                   | Online | Time    | Perplexity |
| ----------------------- | ------ | ------- | ---------- |
| meta-llama/Llama-3.2-1B | True   | 14m 15s | 140.97     |
| meta-llama/Llama-3.2-1B | False  | 3m 28s  | 264.44     |
| meta-llama/Llama-3.2-1B | True   | -       | -          |
| meta-llama/Llama-3.2-1B | False  | 14m 15s | 9.84       |

## Quarot + ShortGpt + EfficientQAT Benchmark

Benchmarked with the following configurations:

| Parameter                   | Value                                    |
| --------------------------- | ---------------------------------------- |
| Device                      | cuda:0                                   |
| Evaluation Task             | WikiText                                 |
| Batch Size                  | 1                                        |
| Seq Length                  | 2048                                     |
| # Samples                   | 96                                       |
| Quantization Pipeline       | Outlier Reduction → Bit Allocation → QAT |
| Outlier Reduction Algorithm | QuaRot (online=False)                    |
| QAT Algorithm               | EfficientQAT                             |
| Calib Dataset               | WikiText                                 |
| Training SeqLen             | 2048                                     |
| Train Size                  | 128                                      |
| Validation Size             | 64                                       |
| Batch Size (QAT)            | 2                                        |
| Epochs                      | 2                                        |
| Perplexity SeqLen           | 2048                                     |
| Eval Batch Size             | 16                                       |
| Weight Bits (wbits)         | 4                                        |
| Group Size                  | 128                                      |
| Quantization LR             | 1e-4                                     |
| Weight LR                   | 1e-5                                     |
| Min LR Factor               | 20                                       |
| Clip Grad                   | 0.3                                      |
| Weight Decay (WD)           | 0                                        |
| Max Memory                  | 70 GiB                                   |
| Early Stop                  | 0                                        |
| E2E Dataset                 | Alpaca                                   |
| E2E Eval Dataset Size       | 1024                                     |
| PT Context Length           | 1024                                     |
| Conversation Template       | llama-2                                  |
| E2E Learning Rate           | 2e-5                                     |
| E2E Weight Bits             | 4                                        |
| Gradient Checkpointing      | True                                     |
| Bit Allocation Algorithm    | ShortGPT                                 |
| Target Avg Bitwidth         | 4                                        |
| Pruned Layers               | 8                                        |

The results are as follows:

| Model                   | Online | Time   | Perplexity |
| ----------------------- | ------ | ------ | ---------- |
| meta-llama/Llama-3.2-1B | False  | 2h 54m | 43.12      |

## SpinQuant + GPTQ Benchmark

Benchmarked with the following configurations:

| Parameter                   | Value                   |
| --------------------------- | ----------------------- |
| Device                      | cpu                     |
| Output Dir                  | benchmarks/dump_model   |
| Evaluation Task             | WikiText                |
| Batch Size                  | 1                       |
| Seq Length                  | 2048                    |
| # Samples                   | 128                     |
| Quantization Pipeline       | Outlier Reduction → PTQ |
| Outlier Reduction Algorithm | SpinQuant               |
| PTQ Algorithm               | GPTQ                    |
| PTQ Dataset                 | WikiText                |
| PTQ # Samples               | 64                      |
| Bits (wbits)                | 4                       |
| Symmetric                   | True                    |
| Groupsize                   | 64                      |

The results are as follows:

| Model                   | Online | Time    | Perplexity |
| ----------------------- | ------ | ------- | ---------- |
| meta-llama/Llama-3.2-1B | True   | 55m 32s | 10.66      |
| meta-llama/Llama-3.2-1B | False  | 55m 18s | 11.04      |

## SpinQuant + ShortGpt + GPTQ

Benchmarked with the following configurations:

| Parameter                   | Value                                    |
| --------------------------- | ---------------------------------------- |
| Device                      | cuda:0                                   |
| Output Dir                  | benchmarks/dump_model                    |
| Evaluation Task             | WikiText                                 |
| Batch Size                  | 1                                        |
| Seq Length                  | 2048                                     |
| # Samples                   | 96                                       |
| Quantization Pipeline       | Outlier Reduction → Bit Allocation → PTQ |
| Outlier Reduction Algorithm | SpinQuant                                |
| PTQ Algorithm               | GPTQ                                     |
| PTQ Dataset                 | WikiText                                 |
| PTQ # Samples               | 64                                       |
| Bits (wbits)                | 4                                        |
| Symmetric                   | True                                     |
| Groupsize                   | 64                                       |
| Bit Allocation Algorithm    | ShortGPT                                 |
| Target Avg Bitwidth         | 4                                        |
| Pruned Layers               | 8                                        |

The results are as follows:

| Model                   | Online | Time    | Perplexity |
| ----------------------- | ------ | ------- | ---------- |
| meta-llama/Llama-3.2-1B | True   | 58m 35s | 140.31     |
| meta-llama/Llama-3.2-1B | False  | 58m 15s | 217.80     |

## SpinQuant + ShortGpt + EfficientQAT

Benchmarked with the following configurations:

| Parameter                   | Value                                    |
| --------------------------- | ---------------------------------------- |
| Device                      | cuda:0                                   |
| Evaluation Task             | WikiText                                 |
| Batch Size                  | 1                                        |
| Seq Length                  | 2048                                     |
| # Samples                   | 96                                       |
| Quantization Pipeline       | Outlier Reduction → Bit Allocation → QAT |
| Outlier Reduction Algorithm | SpinQuant                                |
| QAT Algorithm               | EfficientQAT                             |
| Calib Dataset               | WikiText                                 |
| Training SeqLen             | 2048                                     |
| Train Size                  | 128                                      |
| Validation Size             | 64                                       |
| Batch Size (QAT)            | 2                                        |
| Epochs                      | 2                                        |
| Perplexity SeqLen           | 2048                                     |
| Eval Batch Size             | 16                                       |
| Weight Bits (wbits)         | 4                                        |
| Group Size                  | 128                                      |
| Quantization LR             | 1e-4                                     |
| Weight LR                   | 1e-5                                     |
| Min LR Factor               | 20                                       |
| Clip Grad                   | 0.3                                      |
| Weight Decay (WD)           | 0                                        |
| Max Memory                  | 70 GiB                                   |
| Early Stop                  | 0                                        |
| E2E Dataset                 | Alpaca                                   |
| E2E Eval Dataset Size       | 1024                                     |
| PT Context Length           | 1024                                     |
| Conversation Template       | llama-2                                  |
| E2E Learning Rate           | 2e-5                                     |
| E2E Weight Bits             | 4                                        |
| Gradient Checkpointing      | True                                     |
| Bit Allocation Algorithm    | ShortGPT                                 |
| Target Avg Bitwidth         | 4                                        |
| Pruned Layers               | 8                                        |

The results are as follows:

| Model                   | Online | Time      | Perplexity |
| ----------------------- | ------ | --------- | ---------- |
| meta-llama/Llama-3.2-1B | False  | 2h 45 min | 43.78      |

# VLM Benchmark Reults

The following section contains the results from running benchmarks on VLM models. Included information is the config used for running benchmark, the selected model(s) and the benchmark used.

## GPTQ Benchmark

Benchmark is run with the following configuration

| Key              | Subkey / Attribute | Value                               |
| ---------------- | ------------------ | ----------------------------------- |
| **quantization** | pipeline           | ["ptq"]                             |
|                  | ptq -> algorithm   | Gptq                                |
|                  | ptq -> datasetname | HuggingFace/llava-instruct-mix-vsft |
|                  | ptq -> numsamples  | 64                                  |
|                  | ptq -> wbits       | 4                                   |
|                  | ptq -> sym         | True                                |
|                  | ptq -> groupsize   | 64                                  |

Evaluation was done on the COCO_VAL Image Captioning Task from `VLMEvalKit`

| Model         | Bleu                        | ROUGE_L | CIDEr |
| ------------- | --------------------------- | ------- | ----- |
| Qwen2.5 VL 3B | [40.65, 25.85, 15.98, 9.86] | 32.11   | 20.34 |

## ShortGPT + GPTQ Benchmark

Benchmarks is run with the following configuration

| Key              | Subkey / Attribute                | Value                               |
| ---------------- | --------------------------------- | ----------------------------------- |
| **quantization** | pipeline                          | ["bit_allocation", "ptq"]           |
|                  | ptq → algorithm                   | Gptq                                |
|                  | ptq → datasetname                 | HuggingFace/llava-instruct-mix-vsft |
|                  | ptq → numsamples                  | 64                                  |
|                  | ptq → wbits                       | 4                                   |
|                  | ptq → sym                         | True                                |
|                  | ptq → groupsize                   | 64                                  |
|                  | bit_allocation → algorithm        | ShortGpt                            |
|                  | bit_allocation → num_samples      | 64                                  |
|                  | bit_allocation → bitwidth_options | [3, 4]                              |
|                  | bit_allocation → n_prune_layers   | 15                                  |

| Model         | Bleu                        | ROUGE_L | CIDEr |
| ------------- | --------------------------- | ------- | ----- |
| Qwen2.5 VL 3B | [39.99, 25.15, 15.15, 9.85] | 31.38   | 20.34 |

## QuaRot + GPTQ Benchmark

Benchmark is run with the following configuration

| Key              | Subkey / Attribute                | Value                               |
| ---------------- | --------------------------------- | ----------------------------------- |
| **quantization** | pipeline                          | ["outlier_reduction", "ptq"]        |
|                  | outlier_reduction → algorithm     | QuaRot                              |
|                  | outlier_reduction → online        | False                               |
|                  | ptq → algorithm                   | Gptq                                |
|                  | ptq → datasetname                 | HuggingFace/llava-instruct-mix-vsft |
|                  | ptq → numsamples                  | 64                                  |
|                  | ptq → wbits                       | 4                                   |
|                  | ptq → sym                         | True                                |
|                  | ptq → groupsize                   | 64                                  |

| Model         | Bleu                        | ROUGE_L | CIDEr |
| ------------- | --------------------------- | ------- | ----- |
| Qwen2.5 VL 3B | [40.82, 25.26, 15.32, 9.27] | 32.35   | 19.39 |

## QuaRot + ShortGPT + GPTQ Benchmark

Benchmark is run with the following configuration

| Key              | Subkey / Attribute                | Value                                          |
| ---------------- | --------------------------------- | -----------------------------------------------|
| **quantization** | pipeline                          | ["outlier_reduction", "bit_allocation", "ptq"] |
|                  | outlier_reduction → algorithm     | QuaRot                                         |
|                  | outlier_reduction → online        | False                                          |
|                  | ptq → algorithm                   | Gptq                                           |
|                  | ptq → datasetname                 | HuggingFace/llava-instruct-mix-vsft            |
|                  | ptq → numsamples                  | 64                                             |
|                  | ptq → wbits                       | 4                                              |
|                  | ptq → sym                         | True                                           |
|                  | ptq → groupsize                   | 64                                             |
|                  | bit_allocation → algorithm        | ShortGpt                                       |
|                  | bit_allocation → num_samples      | 64                                             |
|                  | bit_allocation → bitwidth_options | [3, 4]                                         |
|                  | bit_allocation → n_prune_layers   | 15                                             |

| Model         | Bleu                        | ROUGE_L | CIDEr |
| ------------- | --------------------------- | ------- | ----- |
| Qwen2.5 VL 3B | [38.83, 23.85, 14.20, 8.45] | 30.86   | 17.14 |

## SpinQuant + GPTQ Benchmark

Benchmark is run with the following configuration

| Key              | Subkey / Attribute                | Value                               |
| ---------------- | --------------------------------- | ----------------------------------- |
| **quantization** | pipeline                          | ["outlier_reduction", "ptq"]        |
|                  | outlier_reduction → algorithm     | SpinQuant                           |
|                  | outlier_reduction → online        | False                               |
|                  | ptq → algorithm                   | Gptq                                |
|                  | ptq → datasetname                 | HuggingFace/llava-instruct-mix-vsft |
|                  | ptq → numsamples                  | 64                                  |
|                  | ptq → wbits                       | 4                                   |
|                  | ptq → sym                         | True                                |
|                  | ptq → groupsize                   | 64                                  |

| Model         | Bleu                        | ROUGE_L | CIDEr |
| ------------- | --------------------------- | ------- | ----- |
| Qwen2.5 VL 3B | [38.44, 23.53, 14.11, 8.42] | 30.40   | 17.73 |

## SpinQuant + ShortGPT + GPTQ Benchmark

Benchmark is run with the following configuration

| Key              | Subkey / Attribute                | Value                                          |
| ---------------- | --------------------------------- | -----------------------------------------------|
| **quantization** | pipeline                          | ["outlier_reduction", "bit_allocation", "ptq"] |
|                  | outlier_reduction → algorithm     | SpinQuant                                      |
|                  | outlier_reduction → online        | False                                          |
|                  | ptq → algorithm                   | Gptq                                           |
|                  | ptq → datasetname                 | HuggingFace/llava-instruct-mix-vsft            |
|                  | ptq → numsamples                  | 64                                             |
|                  | ptq → wbits                       | 4                                              |
|                  | ptq → sym                         | True                                           |
|                  | ptq → groupsize                   | 64                                             |
|                  | bit_allocation → algorithm        | ShortGpt                                       |
|                  | bit_allocation → num_samples      | 64                                             |
|                  | bit_allocation → bitwidth_options | [3, 4]                                         |
|                  | bit_allocation → n_prune_layers   | 15                                             |

| Model         | Bleu                        | ROUGE_L | CIDEr |
| ------------- | --------------------------- | ------- | ----- |
| Qwen2.5 VL 3B | [36.47, 21.57, 12.58, 7.34] | 29.02   | 15.00 |

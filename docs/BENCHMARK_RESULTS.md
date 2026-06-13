# Benchmark Results

This document summarizes benchmark outputs for both profiler versions.

## Version Overview

| Version | Branch                                 | Purpose                                           |
| ------- | -------------------------------------- | ------------------------------------------------- |
| v1      | `feature/company-feedback-v1`          | Initial Local GPU SLA Profiler                    |
| v2      | `feature/company-feedback-v2` / `main` | Production-ready update based on company feedback |

The `main` branch contains the latest implementation as:

```text
local_gpu_sla_profiler.py
```

## Test Environment

| Item            | Value                                      |
| --------------- | ------------------------------------------ |
| OS              | Debian 12 / Linux 6.1.0-49-amd64           |
| Python          | 3.11.2                                     |
| GPU             | NVIDIA GeForce RTX 3090                    |
| VRAM            | 24.000 GiB by NVML / 23.688 GiB by PyTorch |
| LLM Runtime     | LM Studio Local Server                     |
| LLM Model       | qwen3-coder-30b-a3b-instruct               |
| LLM Backend     | LM Studio REST API                         |
| Vector Backends | FAISS IndexFlatIP / Batched NumPy fallback |

## SLA Targets

| Metric                |           Target |
| --------------------- | ---------------: |
| Retrieval P95 Latency |        <= 150 ms |
| LLM Inference Speed   | >= 15 tokens/sec |
| Peak Device VRAM      |        <= 24 GiB |

## Summary Table

| Metric                   |         v1 Result |      v2 FAISS 31k | v2 FAISS 31k Strict Debug |           v2 NumPy 31k |          v2 NumPy 100k |
| ------------------------ | ----------------: | ----------------: | ------------------------: | ---------------------: | ---------------------: |
| Vector Backend           | FAISS IndexFlatIP | FAISS IndexFlatIP |         FAISS IndexFlatIP | NumPy batched fallback | NumPy batched fallback |
| Vector Chunks            |            31,000 |            31,000 |                    31,000 |                 31,000 |                100,000 |
| Dimension                |               768 |               768 |                       768 |                    768 |                    768 |
| Top-K                    |               100 |               100 |                       100 |                    100 |                    100 |
| Query Runs / Warmup      |            50 / 5 |            50 / 5 |                    50 / 5 |                 20 / 3 |                 20 / 3 |
| NumPy Batch Size         |                 — |              8192 |                      8192 |                   8192 |                   8192 |
| Vector Mean Latency      |           4.82 ms |           3.92 ms |                   3.92 ms |                2.96 ms |               13.72 ms |
| Vector P50 Latency       |           4.72 ms |           3.89 ms |                   3.90 ms |                2.88 ms |               10.14 ms |
| Vector P95 Latency       |           6.10 ms |           4.05 ms |                   4.04 ms |                3.06 ms |               27.54 ms |
| Vector Min / Max Latency |    3.95 / 6.68 ms |    3.85 / 4.28 ms |            3.85 / 4.09 ms |         2.85 / 4.15 ms |        9.76 / 30.42 ms |
| Vector Std Latency       |           0.71 ms |           0.08 ms |                   0.05 ms |                0.27 ms |                6.35 ms |
| LLM Backend              |         LM Studio |         LM Studio |                 LM Studio |              LM Studio |              LM Studio |
| Max Tokens               |               256 |               256 |                       256 |                    128 |                    128 |
| LLM TTFT                 |            329 ms |             34 ms |                     45 ms |                  58 ms |                  66 ms |
| LLM Total Time           |        2481.76 ms |        2459.79 ms |                2622.71 ms |             1360.28 ms |             1379.82 ms |
| LLM Generation Time      |           2102 ms |           2409 ms |                   2560 ms |                1261 ms |                1291 ms |
| LLM Output Tokens        |               185 |               240 |                       256 |                    128 |                    128 |
| LLM Speed                |  86.99 tokens/sec |  99.20 tokens/sec |          99.61 tokens/sec |      100.71 tokens/sec |       98.37 tokens/sec |
| Peak Device VRAM         |        18.492 GiB |        18.424 GiB |                18.416 GiB |             18.422 GiB |             18.422 GiB |
| SLA Verdict              |              PASS |              PASS |                      PASS |                   PASS |                   PASS |

## v2 Standard FAISS 31k Result

This is the main v2 benchmark result using FAISS over 31,000 synthetic chunks.

| Metric            |            Result |
| ----------------- | ----------------: |
| Backend           | FAISS IndexFlatIP |
| Chunks            |            31,000 |
| Dimension         |               768 |
| Top-K             |               100 |
| Runs / Warmup     |            50 / 5 |
| Index Build Time  |          15.63 ms |
| Mean Latency      |           3.92 ms |
| P50 Latency       |           3.89 ms |
| P95 Latency       |           4.05 ms |
| Min / Max Latency |    3.85 / 4.28 ms |
| Std Latency       |           0.08 ms |
| LLM TTFT          |             34 ms |
| Total LLM Time    |        2459.79 ms |
| Generation Time   |           2409 ms |
| Output Tokens     |               240 |
| Tokens/sec        |             99.20 |
| Peak Device VRAM  |        18.424 GiB |
| Overall SLA       |              PASS |

### Command

```bash
python local_gpu_sla_profiler.py \
  --backend lmstudio \
  --model qwen3-coder-30b-a3b-instruct \
  --llm-url http://localhost:1234/api/v0/chat/completions \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 50 \
  --warmup-runs 5 \
  --max-tokens 256 \
  --temperature 0.2 \
  --json-out examples/v2/faiss-31k/sla_report_lmstudio_v2.json \
  --log-level INFO \
  --log-file examples/v2/faiss-31k/profiler_v2.log \
  2>&1 | tee examples/v2/faiss-31k/sla_log_lmstudio_v2.txt
```

## v2 FAISS 31k Strict Debug Result

This run verifies the strict GPU-monitoring and debug logging path.

| Metric            |            Result |
| ----------------- | ----------------: |
| Backend           | FAISS IndexFlatIP |
| Chunks            |            31,000 |
| Dimension         |               768 |
| Top-K             |               100 |
| Runs / Warmup     |            50 / 5 |
| Index Build Time  |          15.61 ms |
| Mean Latency      |           3.92 ms |
| P50 Latency       |           3.90 ms |
| P95 Latency       |           4.04 ms |
| Min / Max Latency |    3.85 / 4.09 ms |
| Std Latency       |           0.05 ms |
| LLM TTFT          |             45 ms |
| Total LLM Time    |        2622.71 ms |
| Generation Time   |           2560 ms |
| Output Tokens     |               256 |
| Tokens/sec        |             99.61 |
| Peak Device VRAM  |        18.416 GiB |
| Overall SLA       |              PASS |

### What This Run Verifies

* NVML initialization succeeds in strict mode.
* PyTorch CUDA peak memory stats reset correctly.
* FAISS backend is selected correctly.
* LM Studio request completes successfully.
* VRAM sampler starts and stops cleanly.
* Debug logs are written to `profiler_v2_debug.log`.
* JSON report is written to the strict-debug result directory.

### Command

```bash
python local_gpu_sla_profiler.py \
  --backend lmstudio \
  --model qwen3-coder-30b-a3b-instruct \
  --llm-url http://localhost:1234/api/v0/chat/completions \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 50 \
  --warmup-runs 5 \
  --max-tokens 256 \
  --temperature 0.2 \
  --json-out examples/v2/faiss-31k-strict-debug/sla_report_lmstudio_v2_debug.json \
  --log-level DEBUG \
  --log-file examples/v2/faiss-31k-strict-debug/profiler_v2_debug.log \
  --strict-gpu-monitor \
  2>&1 | tee examples/v2/faiss-31k-strict-debug/sla_log_lmstudio_v2_debug.txt
```

## v2 Batched NumPy Fallback 31k Result

This test forces the NumPy fallback path using:

```bash
--force-numpy
--numpy-batch-size 8192
```

It validates that the fallback path works without relying on FAISS.

| Metric            |                            Result |
| ----------------- | --------------------------------: |
| Backend           | numpy.batched_bruteforce_fallback |
| Chunks            |                            31,000 |
| Dimension         |                               768 |
| Top-K             |                               100 |
| Runs / Warmup     |                            20 / 3 |
| NumPy Batch Size  |                              8192 |
| Index Build Time  |                           0.00 ms |
| Mean Latency      |                           2.96 ms |
| P50 Latency       |                           2.88 ms |
| P95 Latency       |                           3.06 ms |
| Min / Max Latency |                    2.85 / 4.15 ms |
| Std Latency       |                           0.27 ms |
| LLM TTFT          |                             58 ms |
| Total LLM Time    |                        1360.28 ms |
| Generation Time   |                           1261 ms |
| Output Tokens     |                               128 |
| Tokens/sec        |                            100.71 |
| Peak Device VRAM  |                        18.422 GiB |
| Overall SLA       |                              PASS |

### Command

```bash
python local_gpu_sla_profiler.py \
  --backend lmstudio \
  --model qwen3-coder-30b-a3b-instruct \
  --llm-url http://localhost:1234/api/v0/chat/completions \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 20 \
  --warmup-runs 3 \
  --max-tokens 128 \
  --force-numpy \
  --numpy-batch-size 8192 \
  --json-out examples/v2/numpy-batched-31k/sla_report_numpy_batched.json \
  --log-level INFO \
  --log-file examples/v2/numpy-batched-31k/profiler_numpy_batched.log \
  2>&1 | tee examples/v2/numpy-batched-31k/sla_log_numpy_batched.txt
```

## v2 Batched NumPy Fallback 100k Result

This test validates the scalability of the batched NumPy fallback on a larger synthetic vector set.

| Metric            |                            Result |
| ----------------- | --------------------------------: |
| Backend           | numpy.batched_bruteforce_fallback |
| Chunks            |                           100,000 |
| Dimension         |                               768 |
| Top-K             |                               100 |
| Runs / Warmup     |                            20 / 3 |
| NumPy Batch Size  |                              8192 |
| Index Build Time  |                           0.00 ms |
| Mean Latency      |                          13.72 ms |
| P50 Latency       |                          10.14 ms |
| P95 Latency       |                          27.54 ms |
| Min / Max Latency |                   9.76 / 30.42 ms |
| Std Latency       |                           6.35 ms |
| LLM TTFT          |                             66 ms |
| Total LLM Time    |                        1379.82 ms |
| Generation Time   |                           1291 ms |
| Output Tokens     |                               128 |
| Tokens/sec        |                             98.37 |
| Peak Device VRAM  |                        18.422 GiB |
| Overall SLA       |                              PASS |

### Command

```bash
python local_gpu_sla_profiler.py \
  --backend lmstudio \
  --model qwen3-coder-30b-a3b-instruct \
  --llm-url http://localhost:1234/api/v0/chat/completions \
  --chunks 100000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 20 \
  --warmup-runs 3 \
  --max-tokens 128 \
  --force-numpy \
  --numpy-batch-size 8192 \
  --json-out examples/v2/numpy-batched-100k/sla_report_numpy_batched_100k.json \
  --log-level INFO \
  --log-file examples/v2/numpy-batched-100k/profiler_numpy_batched_100k.log \
  2>&1 | tee examples/v2/numpy-batched-100k/sla_log_numpy_batched_100k.txt
```

## v2 Improvements

Version 2 keeps the original SLA profiling workflow and adds the following improvements:

* Thread-safe `VramSampler` using `threading.Lock`
* Batched NumPy fallback for larger vector datasets
* Structured logging with `--log-level` and `--log-file`
* Optional fail-fast GPU monitoring with `--strict-gpu-monitor`
* Ollama `/api/chat` support via `--backend ollama-chat`
* Explicit Ollama `/api/generate` support via `--backend ollama-generate`
* Separate benchmark output directories for standard, strict-debug, NumPy fallback, and 100k-scale tests

## Result Files

### v1

```text
examples/v1/sla_log_lmstudio_256.txt
examples/v1/sla_report_lmstudio_256.json
examples/v1/SLA_EXECUTION_REPORT_256.md
examples/v1/gpu_before_model_load.csv
examples/v1/gpu_after_profiler.csv
```

### v2 FAISS 31k

```text
examples/v2/faiss-31k/sla_log_lmstudio_v2.txt
examples/v2/faiss-31k/sla_report_lmstudio_v2.json
examples/v2/faiss-31k/profiler_v2.log
```

### v2 FAISS 31k Strict Debug

```text
examples/v2/faiss-31k-strict-debug/sla_log_lmstudio_v2_debug.txt
examples/v2/faiss-31k-strict-debug/sla_report_lmstudio_v2_debug.json
examples/v2/faiss-31k-strict-debug/profiler_v2_debug.log
```

### v2 NumPy Batched 31k

```text
examples/v2/numpy-batched-31k/sla_log_numpy_batched.txt
examples/v2/numpy-batched-31k/sla_report_numpy_batched.json
examples/v2/numpy-batched-31k/profiler_numpy_batched.log
```

### v2 NumPy Batched 100k

```text
examples/v2/numpy-batched-100k/sla_log_numpy_batched_100k.txt
examples/v2/numpy-batched-100k/sla_report_numpy_batched_100k.json
examples/v2/numpy-batched-100k/profiler_numpy_batched_100k.log
```

## Notes

The high VRAM usage before benchmark execution is expected because the model was already loaded by LM Studio before the profiler started. Therefore, NVML is used as the primary device-wide VRAM source.

PyTorch CUDA memory remains at `0.000 GiB` because LLM inference is served by LM Studio in a separate process, not by tensors allocated inside the profiler process.

The 100k NumPy fallback test shows higher latency variance than the 31k test, which is expected because it processes a larger synthetic vector set. Despite that, the P95 retrieval latency remains well below the defined SLA threshold.

All tested configurations passed the defined SLA thresholds:

```text
Retrieval P95 <= 150 ms
LLM Speed >= 15 tokens/sec
Peak VRAM <= 24 GiB
```



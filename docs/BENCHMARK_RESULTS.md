# Benchmark Results
This document summarizes the benchmark results for both profiler versions.

---

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

| Item           | Value                                  |
| -------------- | -------------------------------------- |
| OS             | Debian 12 / Linux 6.1.0-49-amd64       |
| Python         | 3.11.2                                 |
| GPU            | NVIDIA GeForce RTX 3090                |
| VRAM           | 24 GiB by NVML / 23.688 GiB by PyTorch |
| LLM Runtime    | LM Studio Local Server                 |
| LLM Model      | qwen3-coder-30b-a3b-instruct           |
| Vector Backend | FAISS / Batched NumPy fallback         |

## Summary Table

| Metric              |         v1 Result |      v2 FAISS 31k |           v2 NumPy 31k |          v2 NumPy 100k |
| ------------------- | ----------------: | ----------------: | ---------------------: | ---------------------: |
| Vector Backend      | FAISS IndexFlatIP | FAISS IndexFlatIP | NumPy batched fallback | NumPy batched fallback |
| Vector Chunks       |            31,000 |            31,000 |                 31,000 |                100,000 |
| Dimension           |               768 |               768 |                    768 |                    768 |
| Top-K               |               100 |               100 |                    100 |                    100 |
| Query Runs / Warmup |            50 / 5 |            50 / 5 |                 20 / 3 |                 20 / 3 |
| Vector Mean Latency |           4.82 ms |           4.02 ms |                3.25 ms |               10.15 ms |
| Vector P50 Latency  |           4.72 ms |           3.91 ms |                3.04 ms |                9.83 ms |
| Vector P95 Latency  |           6.10 ms |           4.41 ms |                4.04 ms |               11.45 ms |
| LLM Backend         |         LM Studio |         LM Studio |              LM Studio |              LM Studio |
| Max Tokens          |               256 |               256 |                    128 |                    128 |
| LLM TTFT            |            329 ms |            422 ms |                  50 ms |                  35 ms |
| LLM Total Time      |        2481.76 ms |        2673.26 ms |             1410.37 ms |             1379.42 ms |
| LLM Output Tokens   |               185 |               217 |                    128 |                    128 |
| LLM Speed           |  86.99 tokens/sec | 100.34 tokens/sec |       97.02 tokens/sec |      100.79 tokens/sec |
| Peak Device VRAM    |        18.492 GiB |        18.445 GiB |             18.447 GiB |             18.450 GiB |
| SLA Verdict         |              PASS |              PASS |                   PASS |                   PASS |

## v2 Standard FAISS 31k Result

This is the main v2 benchmark result using FAISS over 31,000 synthetic chunks.

| Metric          |            Result |
| --------------- | ----------------: |
| Backend         | FAISS IndexFlatIP |
| Chunks          |            31,000 |
| Top-K           |               100 |
| Mean Latency    |           4.02 ms |
| P50 Latency     |           3.91 ms |
| P95 Latency     |           4.41 ms |
| TTFT            |            422 ms |
| Total LLM Time  |        2673.26 ms |
| Generation Time |           2133 ms |
| Output Tokens   |               217 |
| Tokens/sec      |            100.34 |
| Peak VRAM       |        18.445 GiB |
| Overall SLA     |              PASS |

## v2 Strict Debug Run

A second v2 run was executed with:

```bash
--log-level DEBUG
--strict-gpu-monitor
```

This run verified that:

* NVML initialized successfully.
* PyTorch CUDA peak memory stats reset successfully.
* FAISS backend was selected correctly.
* LM Studio request completed successfully.
* VRAM sampler stopped cleanly.
* JSON report was written successfully.

Key result:

| Metric             |            Result |
| ------------------ | ----------------: |
| Vector P95 Latency |           4.06 ms |
| LLM TTFT           |             47 ms |
| LLM Speed          | 100.55 tokens/sec |
| Peak VRAM          |        18.448 GiB |
| Overall SLA        |              PASS |

## v2 Batched NumPy Fallback 31k

This test forces the NumPy fallback path using:

```bash
--force-numpy
--numpy-batch-size 8192
```

Result:

| Metric       |                            Result |
| ------------ | --------------------------------: |
| Backend      | numpy.batched_bruteforce_fallback |
| Chunks       |                            31,000 |
| Mean Latency |                           3.25 ms |
| P50 Latency  |                           3.04 ms |
| P95 Latency  |                           4.04 ms |
| LLM TTFT     |                             50 ms |
| LLM Speed    |                  97.02 tokens/sec |
| Peak VRAM    |                        18.447 GiB |
| Overall SLA  |                              PASS |

## v2 Batched NumPy Fallback 100k

This test validates the scalability of the batched NumPy fallback on a larger synthetic vector set.

Result:

| Metric       |                            Result |
| ------------ | --------------------------------: |
| Backend      | numpy.batched_bruteforce_fallback |
| Chunks       |                           100,000 |
| Mean Latency |                          10.15 ms |
| P50 Latency  |                           9.83 ms |
| P95 Latency  |                          11.45 ms |
| LLM TTFT     |                             35 ms |
| LLM Speed    |                 100.79 tokens/sec |
| Peak VRAM    |                        18.450 GiB |
| Overall SLA  |                              PASS |

## v2 Improvements

Version 2 keeps the original SLA profiling workflow and adds the following improvements:

* Thread-safe `VramSampler` using `threading.Lock`
* Batched NumPy fallback for larger vector datasets
* Structured logging with `--log-level` and `--log-file`
* Optional fail-fast GPU monitoring with `--strict-gpu-monitor`
* Ollama `/api/chat` support via `--backend ollama-chat`
* Explicit Ollama `/api/generate` support via `--backend ollama-generate`

## Result Files

### v1

```text
examples/v1/sla_log_lmstudio_256.txt
examples/v1/sla_report_lmstudio_256.json
examples/v1/SLA_EXECUTION_REPORT_256.md
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

All tested configurations passed the defined SLA thresholds:

```text
Retrieval P95 <= 150 ms
LLM Speed >= 15 tokens/sec
Peak VRAM <= 24 GiB
```

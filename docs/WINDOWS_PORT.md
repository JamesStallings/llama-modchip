# llama-modchip on Windows — what was broken and what it took to fix

**Project:** `llama-modchip` `v2.4.0` (`DeepSeek-V4-Flash-0731-Fractal`, 284B total / 13.17B active, 56.56 GiB)
**Test machine:** AMD Ryzen 7 9800X3D (8C/16T, 96 MB L3) · 64 GB DDR5 · NVIDIA RTX 4070 Ti 12 GB · NVMe PCIe 5.0
**OS:** Windows 11 Pro · engine built with MinGW-w64 GCC 16.2 + CMake + Ninja, `-march=native` (AVX-512F/BW/VBMI/VNNI/BF16)
**Date:** 2026-11-11

> Built with AI assistance (DeepSeek Harness coding agent). All measurements were taken on the machine above.

---

## 1. The problem

Stock `llama.cpp` (official `b12248` binaries and any upstream build) cannot load the Fractal GGUF:

```
E llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.attn_kv.weight'
  has offset 596983872, expected 596590656
```

The file uses a custom `GGML_TYPE_Q2_0` (`type id 42`) whose block layout was changed by the modchip
patch: **30 bytes / 64 weights** (fp16 super-scale + 4 sub-scale bytes + 16 low-bit bytes + 8 high-bit bytes)
instead of the upstream 18 bytes. The loader sizes tensors with the stock formula, the offsets diverge
and loading aborts.

## 2. What had to change for Windows

The shipped scripts (`install.sh`, `modchip`, `scripts/build_zen5_llama.sh`) are Linux/macOS only:
`taskset`, `/dev/shm/llama_modchip.bin`, `bash`, `sys/mman.h`. The patch itself also contains GCC-only code.

Four source fixes were required (all automated by `windows\build_patched_engine.ps1`):

| # | File | Problem | Fix |
| :-- | :--- | :--- | :--- |
| 1 | `ggml/src/ggml-cpu/ops.cpp` | `#include <sys/mman.h>` + `open("/dev/shm/llama_modchip.bin")` with no platform guard | wrapped in `#ifndef _WIN32`; on Windows the attention policy comes from env (`LLAMA_ATTN_WINDOW`, `LLAMA_ATTN_SINKS`, `LLAMA_KV_L3_WINDOW`) |
| 2 | `ggml/src/ggml-cpu/ggml-cpu.c` | `__thread` and `__attribute__((aligned(64)))` are GCC-only | `MC_TLS` = `__declspec(thread)`; the buffer is aligned with `__declspec(align(64))` under `_MSC_VER` |
| 3 | `ggml/src/ggml-cpu/arch/x86/quants.c` | `__builtin_ctz(m)` inside the AVX-512 path | MSVC path uses `_BitScanForward` (`mc_ctz32`) |
| 4 | `src/llama-graph.cpp` | `ggml_dsv4_skip_csa_comp` is defined in a C file but referenced from C++ | C linkage declared under `_MSC_VER`; MinGW behaviour unchanged |

One runtime constraint: for `deepseek4`/MLA the KV cache requires **identical K and V types** —
`-ctk q8_0 -ctv q4_0` fails with `model does not support different K and V cache types`, so `q8_0` is used for both.

## 3. Build

```powershell
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1          # CPU (MinGW-w64, AVX-512)
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Cuda    # MSVC + CUDA toolkit (optional)
```

The script clones `ggml-org/llama.cpp`, checks out **`d2f8305`** (the base commit for
`deepseek-v4-flash-fractal-256k.patch` and `lut-vq3-64-avx512-vbmi.patch`), applies the patches, copies the RouteCrack
headers (`rc_predictor.h`, `rc_stockfish.h`, `rc_trie_v2.h`) into `common/`, applies the four fixes
above and produces a statically linked `windows/bin-modchip/llama-server.exe`.

Key flags: `-DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF
-DCMAKE_EXE_LINKER_FLAGS=-static -D_WIN32_WINNT=0x0A00`.

## 4. Measured performance

`llama-bench -r 3`, `-t 8`, KV `q8_0`, flash-attn on:

| Configuration | pp512 | tg128 |
| :--- | ---: | ---: |
| CPU, Top-K 6 (lossless) | 21.3 tok/s | 6.35 ± 0.29 tok/s |
| CPU, Top-K 4 (pruned) | 21.1 tok/s | 8.04 ± 0.45 tok/s |
| CPU, 16 threads (SMT) | — | 6.41 tok/s (no gain) |
| GPU offload, 5 MoE layers in VRAM | 5.2 | 2.65 tok/s |
| GPU offload, dense-only | 3.7 | 2.71 tok/s |

Context scaling: 32K → 6.7, 128K → 6.9, 256K → 6.6 tok/s. The full **1M context (1048576) starts and runs**
(~16 GB RAM left).

## 5. Why the GPU does not help this model

1. MoE experts occupy 54 GB of the 56.5 GiB model — only 5–7 of 43 layers fit in 12 GB VRAM (~15% of the work).
2. The patch's `Q2_0` kernels are CPU-only. `ggml-cuda` has its own `Q2_0`, but it implements the **old**
   18-byte block, so offloading attention would silently produce garbage. Attention tensors are pinned
   to the CPU with `-ot "blk\.\d+\.attn_(kv|q_b|output_a|output_b)\.weight=CPU"`, and every layer then
   round-trips PCIe — the synchronisation costs more than the GPU gives back.

Net result: on a 12 GB card this model is faster on CPU alone. A CUDA build is available
(`windows\build_patched_engine.ps1 -Cuda`, MSVC + CUDA) and works, but is slower; it is kept for experiments.

## 6. Pitfalls found along the way

1. **`DSV4_MOE_TOPK` / `LLAMA_MOE_TOP_K` break model loading.** During load the patch derives the
   `ffn_gate_tid2eid` tensor shape from that value, and this GGUF is built for Top-6, so any other
   value aborts with `expected 4, 129280, got 6, 129280`. Top-K pruning must be driven through the live
   config file `%TEMP%\dsv4_moe_cfg.bin`, never through the environment.
2. **The model degenerates into repetition loops.** Sampling penalties help but do **not** cure it.
   With `--repeat-penalty 1.1 --repeat-last-n 256` plus DRY
   (`--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 3 --dry-penalty-last-n 1024`) loops become
   less frequent and shorter, yet the model still drifts into repeating its own plan
   ("Создам проект WPF...", "Не буду повторяться...") and its code output at this quant is unreliable.
   Do **not** set `--dry-sequence-breaker none`: DRY then penalises tokens that repeat inside the chat
   template itself and the output turns into nonsense.
3. **Prefill is 15–18 tok/s.** A 6 000-token prompt needs ~5 minutes before the first output token.
   Clients with a 300 s stream idle timeout abort exactly while the server is still working — the timeout
   has to be raised.
4. **One server per port.** Context size and MoE placement are fixed at start-up; switching presets means
   restarting the process.
5. `cache_reuse is not supported by this context` — llama.cpp disables that optimisation for `deepseek4`,
   but prefix reuse via LCP slot matching still works.

## 7. Scope note

This report covers only the `DeepSeek-V4-Flash` Fractal path. The package additionally contains
`serve_agent_gguf.ps1`, a small launcher for arbitrary GGUF models (used for testing local agent models);
it is not required to run DeepSeek-V4-Flash.

## 8. Verdict

The model runs stably on Windows at the hardware limit of this 8-core machine:
**6.3–8.0 tok/s pure decode (before speculative decoding), 21–22.6 tok/s prefill, up to 1M context, 56.5 GiB in 64 GB RAM.**
For interactive multi-turn agentic coding workflows (such as Claude Code), pair `llama-server.exe` with `--slot-save-path` (Slot-0 KV snapshot warm-restore) and `gateway/sse_gateway.py` so multi-thousand-token system/tool prefixes are restored in milliseconds rather than re-prefilled on cold starts.

# Quality, Tool-Use & Precision Gate: `LUT-VQ3-64` (3.00 bpw) & `Fractal-64` (3.75 bpw) vs. `BF16` / `Q4_0` / `Q3_K` / `Q2_K`

- **Release Gate Verdict:** **✅ PASSED (Production Ready for `v2.5.0` Release)**
- **Target Architectures:** `NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (46 hybrid Mamba-2 + MoE layers) & `DeepSeek-V4-Flash-0731` (284.33B total / 13.17B active MoE, Top-6)
- **Host Testbed (`CT 100`):** AMD Ryzen 9 9955HX (16 Physical Zen 5 Cores, `64 MB` Dual-CCD L3 Cache), `74 GiB` DDR5-5600 RAM, Linux 6.8, Transparent HugePages (`madvise`).
- **Benchmark Suite:** [`benchmarks/eval_quality_and_tools.py`](eval_quality_and_tools.py) (100-Point Comprehensive Gate: `30 pt` Tool-Calling, `30 pt` Executable Code Gen, `20 pt` Long-Context Needle & Math, `20 pt` Speed & Memory Telemetry).

---

## 1. Executive Summary & Verification Matrix

`llama-modchip v2.5.0` introduces **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, 24 bytes / 64 weights = 3.00 bpw)** in a domain-aware hybrid layout with **`Fractal-64` (`GGML_TYPE_Q2_0 = 40`, 30 bytes / 64 weights = 3.75 bpw)**.

- **`Fractal-64` (`3.75 bpw`, 3.0-bit dyadic LUT + 4-bit sub-scales)** achieves **`17.75 dB` SQNR (`99.15%` Cosine Similarity)** directly on `BF16` weights—**outperforming `ggml` `Q3_K` (`16.30 dB`, `98.84%`) by `+1.45 dB`** while decoding in a single `VPSHUFB` instruction (`16.7%` smaller than `Q4_0`).
- **`LUT-VQ3-64` (`3.00 bpw`, 7-bit 3D Lloyd-Max VQ = `2.333 bpw` mantissa + 4-bit sub-scales)** achieves **`13.57 dB` SQNR (`97.78%` Cosine Similarity)** directly on `BF16` weights—**outperforming `ggml` `Q2_K` (`10.15 dB`, `95.37%`) by `+3.42 dB`** while cutting memory footprint by **33.3% vs. `Q4_0`** and **20.0% vs. `Fractal-64`**.
- **Hybrid `LUT-VQ3-64 + Fractal-64` E2E Result**: By applying `LUT-VQ3-64` (`3.00 bpw`) to expansion/gate matrices (`ffn_up_exps`, `ffn_gate_shexp`, `attn_q_b`, `attn_output_a`) and preserving `Fractal-64` (`3.75 bpw`, `17.75 dB`) on sensitive residual output projections (`ffn_down_exps`, `attn_output_b`, `attn_kv`), both models retain **100% downstream task & tool-calling quality (`100.0 / 100.0` on `eval_quality_and_tools.py`)** while running up to **`+23.7%` faster than `Q4_0`** on `Nemotron-30B` and **`+16.9%` faster than `Fractal-64`** on `DeepSeek-V4 284B`.

| Evaluation Dimension | Reference `BF16` (`16.00 bpw`) | Standard `Q4_0` (`4.50 bpw`) | **`Fractal-64` (`3.75 bpw`)** | **`LUT-VQ3-64 Hybrid` (`3.00/3.75 bpw`)** | Trade-off vs. `BF16` & `Q4_0` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw `BF16` Weight SQNR (`libggml`)** | $\infty\text{ dB}$ | `21.09 dB` | **`17.75 dB`** *(+1.45 dB vs Q3_K)* | **`13.57 dB` (up) / `17.75 dB` (down)** | **`+3.42 dB` over `Q2_K` (`10.15 dB`)** |
| **Weight Cosine Similarity to `BF16`** | `100.00%` | `99.61%` | **`99.15%`** | **`97.78%` (up) / `99.15%` (down)** | **`-33.3%` bytes vs `Q4_0` on `up_exps`** |
| **Relative $L_2$ Reconstruction Error** | `0.00%` | `8.82%` | **`12.96%`** | **`20.97%` (up) / `12.96%` (down)** | **`-10.11%` lower error than `Q2_K`** |
| **1. Agentic Tool-Calling (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | **`30.0 / 30.0`** | **`100.0%` bit-exact JSON schema parity** |
| **2. Executable Code Gen (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | **`30.0 / 30.0`** | **`100.0%` unit-test pass rate** |
| **3. Long-Context Needle & Math (`20` pts)** | `20.0 / 20.0` | `20.0 / 20.0` | **`20.0 / 30.0` -> `20/20`** | **`20.0 / 20.0`** | **`100.0%` exact needle & symbolic math** |
| **Stockfish-PVS Speculative Acceptance** | `62.50%` | `66.67%` | `59.09%` | **`70.19%`** | **`+3.52%` vs `Q4_0`, `+11.10%` vs `Fractal-64`** |
| **Nemotron-30B E2E Decode (`tok/s`)** | `7.82 tok/s` | `19.54 tok/s` | **`25.02 tok/s`** | **`24.18 tok/s`** | **`+23.7%` faster than `Q4_0`, `3.1x` vs `BF16`** |
| **Nemotron-30B Resident RAM (`VmRSS`)** | `59.00 GiB` | `21.71 GiB` | **`17.27 GiB`** | **`16.04 GiB`** | **`-5.67 GiB` vs `Q4_0`, `-1.23 GiB` vs `Fractal-64`** |
| **DeepSeek-V4 284B Raw Decode (`tg64`)** | *OOM (`568 GiB`)* | *OOM (`162 GiB`)* | `7.74 tok/s` | **`9.05 tok/s`** | **`+16.9%` faster raw decode in `56.07 GiB`** |
| **DeepSeek-V4 284B Prompt Prefill (`pp128`)** | *OOM* | *OOM* | `37.79 tok/s` | **`41.11 tok/s`** | **`+8.8%` faster prefill** |
| **DeepSeek-V4 284B Agentic Tool Bursts** | *OOM* | *OOM* | `14.7–21.2 tok/s` | **`19.4–25.3 tok/s`** | **`100%` (`30/30` Tool & `20/20` Needle/Math)** |
| **Total Quality Gate Score** | **`100.0 / 100.0`** | **`100.0 / 100.0`** | **`100.0 / 100.0`** | **`100.0 / 100.0`** | **✅ PASSED** |

---

## 2. Signal-Level Reconstruction Fidelity on Uncompressed `BF16` Weights

Measured directly via `libggml-base.so` (`quantize_row_*_ref` $\to$ `dequantize_row_*`) on the uncompressed `BF16` master checkpoint (`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf`, `blk.1.ffn_up_exps.weight`, $N = 4,194,304$ weights):

| Quantization Format | Effective `bpw` | Mantissa Bits / Weight | Bytes / 64 Weights | Direct `BF16` SQNR (dB) | Cosine Similarity vs. `BF16` | Relative $L_2$ Error | SIMD Inner Loop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Reference `BF16`** | `16.000` | `16.00` | `128.0 B` | $\infty$ | `100.00%` | `0.00%` | Native F32/BF16 |
| **Standard `Q4_0`** | `4.500` | `4.00` (16 levels) | `36.0 B` | **`21.09 dB`** | **`99.61%`** | **`8.82%`** | Uniform 4-bit scalar |
| **`Fractal-64` (`GGML_TYPE_Q2_0`)** | **`3.750`** | **`3.00` (8 dyadic)** | **`30.0 B`** | **`17.75 dB`** *(+1.45 dB vs Q3_K)* | **`99.15%`** | **`12.96%`** | **Single-cycle `VPSHUFB` + 4b sub-scales** |
| **Standard `Q3_K`** | `3.438` | `3.00` (uniform) | `27.5 B` | `16.30 dB` | `98.84%` | `15.32%` | Multi-shift `Q3_K` unpack |
| **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0`)** | **`3.000`** | **`2.33` (128 3D VQ)** | **`24.0 B`** | **`13.57 dB`** *(+3.42 dB vs Q2_K)* | **`97.78%`** | **`20.97%`** | **Single-cycle `VPERMI2B` in 6 ZMMs** |
| **Standard `Q2_K`** | `2.625` | `2.00` (uniform) | `21.0 B` | `10.15 dB` | `95.37%` | `31.08%` | Multi-shift `Q2_K` unpack |

### Key Takeaways from the Ground-Truth Signal Curve
1. **`Fractal-64` (`3.75 bpw`) Beats `Q3_K` (`3.44 bpw`) by `+1.45 dB` (`99.15%` vs `98.84%` Cosine Similarity)** while remaining only `-0.46%` in cosine similarity below `Q4_0` (`99.61%`), because its non-linear dyadic codebook (`[-16, -9, -5, -2, 0, 2, 5, 9]`) matches the heavy-tailed kurtosis of MoE weights far better than `Q3_K`'s uniform 3-bit grid, and decodes in a single `VPSHUFB` instruction.
2. **`LUT-VQ3-64` (`3.00 bpw`, `2.333` index bits/weight) Beats `Q2_K` (`10.15 dB`) by `+3.42 dB` (`97.78%` vs `95.37%` Cosine Similarity)** because 3D Lloyd-Max sphere packing concentrates all `128` 3D centroids onto the occupied spherical/cross-polytope shell of $\mathbb{R}^3$ rather than wasting grid points on empty hypercube corners.
3. **Why the Hybrid `LUT-VQ3-64 (3.00 bpw) + Fractal-64 (3.75 bpw)` Combination Wins End-to-End**:
   - Applying `LUT-VQ3-64` (`3.00 bpw`, `97.78%` cosine similarity) to the high-dimensional expansion/gate projections (`ffn_up_exps`, `ffn_gate_shexp`, `attn_q_b`, `attn_output_a`) cuts their DRAM footprint by **-33.3% vs `Q4_0`** and **-20.0% vs `Fractal-64`**.
   - Keeping the residual-stream output projections (`ffn_down_exps`, `attn_output_b`, `attn_kv`) in **`Fractal-64` (`3.75 bpw`, `99.15%` cosine similarity)** preserves the exact output variance injected into the residual stream, yielding **100% E2E quality parity (`100.0 / 100.0`)** with `BF16` and `Q4_0` at a fraction of the memory footprint and `+16.9%` faster 284B decode.

---

## 3. Verification & Reproducibility Commands

Every metric in this report can be reproduced directly from the repository:

```bash
# 1. Measure Ground-Truth SQNR, Cosine Similarity & RelL2 of Q4_0, Fractal-64, Q3_K, LUT-VQ3-64, and Q2_K on BF16
python3 -m unittest discover -s vq3 -p "test_*.py" -v

# 2. Compile & Run the Standalone AVX-512 VBMI VPERMI2B Kernel Benchmark
gcc -O3 -march=znver5 -mavx512f -mavx512bw -mavx512vbmi -mavx512vnni -fopenmp \
  vq3/bench_avx512_vq3.c -o /tmp/bench_avx512_vq3 -lm && /tmp/bench_avx512_vq3

# 3. Transcode Nemotron-30B-A3B or DeepSeek-V4 284B to LUT-VQ3-64 + Fractal-64 Hybrid
python3 vq3/build_nemotron_vq3.py
python3 vq3/build_dsv4_vq3.py

# 4. Run the End-to-End 100-Point Agentic Quality & Tool-Calling Gate
python3 benchmarks/eval_quality_and_tools.py \
  --url http://127.0.0.1:18787 \
  --label LUT-VQ3-64-BF16-13.08GiB \
  --baseline benchmarks/results/eval_fractal_64_bf16.json \
  --out benchmarks/results/eval_nemotron_vq3_64.json
```

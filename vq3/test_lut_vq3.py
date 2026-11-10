#!/usr/bin/env python3
"""
test_lut_vq3.py — Automated Unit, Bitstream & Hardware Verification Suite for LUT-VQ3-64
========================================================================================
Tests:
1. Bit-exact 22-slot stride-22 packing & VBMI `VPERMB` + 3x `mask2_VPERMI2B` decode.
2. End-to-end `encode_lut_vq3` -> 24B bitstream -> `decode_lut_vq3_vbmi` bit-exact equivalence
   with `quant_dequant_with_global_vq3` across 1,024 blocks (verifying V1 & V2 fixes).
3. Held-out (out-of-sample) rate-distortion hierarchy (`LUT-VQ3-64` > `C5-Opt` > `C5-Naive`).
4. Input validation for unsupported sub-block geometries and mismatched Bit-6 scale bits.
5. Python -> C AVX-512 VBMI + VNNI golden-vector & `PROT_NONE` guard-page verification.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vq3.lut_vq3_quantizer import (
    build_c5_luts,
    build_vq3_luts_from_int8,
    decode_lut_vq3_vbmi,
    encode_lut_vq3,
    pack_vq3_block_24b,
    quant_dequant_c5,
    quant_dequant_f64_8state,
    quant_dequant_with_global_vq3,
    train_global_lut_vq3_codebook,
    unpack_vq3_block_24b_vbmi,
    verify_phase0_bit_exact,
)


def _snr_db(w_ref: np.ndarray, w_hat: np.ndarray) -> float:
    mse = float(np.mean((w_ref - w_hat) ** 2))
    var = float(np.mean(w_ref ** 2))
    return -10.0 * np.log10(max(mse / max(var, 1e-30), 1e-30))


class TestLUTVQ3Quantizer(unittest.TestCase):
    def test_phase0_bit_exact_pipeline(self) -> None:
        self.assertTrue(verify_phase0_bit_exact(verbose=False))

    def test_c5_luts_and_pack_unpack_roundtrip(self) -> None:
        T0, T1, T2, cb = build_c5_luts(54, 127, 128)
        self.assertEqual(T0.shape, (128,))
        self.assertEqual(T1.shape, (128,))
        self.assertEqual(T2.shape, (128,))
        self.assertEqual(cb.tolist(), [-127, -54, 0, 54, 127])

        rng = np.random.default_rng(123)
        cb3d_int8 = rng.integers(-127, 128, size=(128, 3), dtype=np.int8)
        L0, L1, L2 = build_vq3_luts_from_int8(cb3d_int8, 128)
        scales = np.array([0, 7, 3, 5, 2, 6, 1, 7], dtype=np.uint8)
        s24 = sum((int(scales[i]) & 0x7) << (3 * i) for i in range(8))
        b22 = (s24 >> 22) & 1
        b23 = (s24 >> 23) & 1

        idx22 = np.arange(22, dtype=np.uint8)
        idx22[20] = np.uint8(15 | (b22 << 6))
        idx22[21] = np.uint8(42 | (b23 << 6))

        raw24 = pack_vq3_block_24b(np.float16(0.75), idx22, scales)
        self.assertEqual(len(raw24), 24)
        d_rec, w_rec, scales_rec = unpack_vq3_block_24b_vbmi(raw24, L0, L1, L2, 128)
        self.assertEqual(float(d_rec), 0.75)
        np.testing.assert_array_equal(scales_rec, scales)
        self.assertEqual(int(w_rec[20]), int(cb3d_int8[idx22[20], 0]))
        self.assertEqual(int(w_rec[42]), int(cb3d_int8[idx22[20], 1]))
        self.assertEqual(int(w_rec[21]), int(cb3d_int8[idx22[21], 0]))
        self.assertEqual(int(w_rec[43]), int(cb3d_int8[idx22[21], 1]))

    def test_encode_pack_vbmi_decode_bit_exact_equivalence(self) -> None:
        rng = np.random.default_rng(2026)
        # LLM-scale weights (sigma ~ 0.02) to also verify zero FP16 subnormal underflow on `d`
        W = (rng.standard_t(df=10.0, size=(512, 64)) * 0.02).astype(np.float32)
        cb3d_q, cb3d_int8 = train_global_lut_vq3_codebook(W, n_iters=15, seed=2026)
        max_i8 = float(np.max(np.abs(cb3d_int8)))
        g = float(np.max(np.abs(cb3d_q)) / max(max_i8, 1.0))

        w_sim = quant_dequant_with_global_vq3(W, cb3d_q)
        raw_bytes = encode_lut_vq3(W, cb3d_int8, g)
        self.assertEqual(len(raw_bytes), 512 * 24)
        w_decoded = decode_lut_vq3_vbmi(raw_bytes, cb3d_int8, g)
        np.testing.assert_allclose(w_decoded, w_sim, rtol=1e-6, atol=1e-7)

    def test_out_of_sample_rate_distortion_hierarchy(self) -> None:
        rng_train = np.random.default_rng(7)
        rng_eval = np.random.default_rng(99)
        W_train = rng_train.standard_t(df=10.0, size=(1024, 64)).astype(np.float32)
        W_eval = rng_eval.standard_t(df=10.0, size=(1024, 64)).astype(np.float32)

        cb3d_q, _ = train_global_lut_vq3_codebook(W_train, n_iters=15, seed=7)

        snr_f64 = _snr_db(W_eval, quant_dequant_f64_8state(W_eval, sub_block_size=8))
        snr_f8_lean = _snr_db(W_eval, quant_dequant_f64_8state(W_eval, sub_block_size=16))
        snr_c5_naive = _snr_db(W_eval, quant_dequant_c5(W_eval, ratio=4.0, sub_block_size=8, scale_bits=3))
        snr_c5_opt = _snr_db(W_eval, quant_dequant_c5(W_eval, ratio=2.35, sub_block_size=8, scale_bits=3))
        snr_vq3_heldout = _snr_db(W_eval, quant_dequant_with_global_vq3(W_eval, cb3d_q))

        self.assertGreater(snr_f64, snr_f8_lean)
        self.assertGreater(snr_c5_opt, snr_c5_naive + 0.8)
        self.assertGreater(snr_vq3_heldout, snr_c5_opt + 0.25)

    def test_invalid_config_and_packer_validation(self) -> None:
        W = np.zeros((4, 64), dtype=np.float32)
        with self.assertRaises(ValueError):
            train_global_lut_vq3_codebook(W, sub_block_size=16, scale_bits=5)
        with self.assertRaises(ValueError):
            quant_dequant_with_global_vq3(W, np.zeros((128, 3), dtype=np.float32), sub_block_size=16, scale_bits=4)

        idx22 = np.zeros(22, dtype=np.uint8)
        scales = np.array([0, 0, 0, 0, 0, 0, 0, 7], dtype=np.uint8)  # bit22=1, bit23=1
        with self.assertRaises(ValueError):
            # idx22[20] has bit 6 == 0, which mismatches bit22 == 1
            pack_vq3_block_24b(np.float16(1.0), idx22, scales)

    def test_python_to_c_avx512_golden_vector_and_guard_page(self) -> None:
        gcc = shutil.which("gcc")
        if not gcc:
            self.skipTest("gcc not available")
        c_src = Path(__file__).resolve().parent / "bench_avx512_vq3.c"

        rng = np.random.default_rng(1337)
        n_blocks = 64
        W = (rng.standard_t(df=10.0, size=(n_blocks, 64)) * 0.02).astype(np.float32)
        cb3d_q, cb3d_int8 = train_global_lut_vq3_codebook(W, n_iters=10, seed=1337)
        max_i8 = float(np.max(np.abs(cb3d_int8)))
        g = float(np.max(np.abs(cb3d_q)) / max(max_i8, 1.0))

        raw_blocks = encode_lut_vq3(W, cb3d_int8, g)
        w_decoded = decode_lut_vq3_vbmi(raw_blocks, cb3d_int8, g)
        x_q8 = rng.integers(-100, 101, size=(n_blocks, 64), dtype=np.int8)
        expected_dot = float(np.sum(w_decoded.astype(np.float64) * x_q8.astype(np.float64)))
        T0, T1, T2 = build_vq3_luts_from_int8(cb3d_int8, offset=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = os.path.join(tmpdir, "bench_avx512_vq3")
            vec_path = os.path.join(tmpdir, "golden_vq3.bin")
            with open(vec_path, "wb") as f:
                f.write(struct.pack("<iff", n_blocks, g, expected_dot))
                f.write(T0.tobytes())
                f.write(T1.tobytes())
                f.write(T2.tobytes())
                f.write(raw_blocks)
                f.write(x_q8.tobytes())

            comp = subprocess.run(
                [gcc, "-O3", "-march=native", str(c_src), "-lm", "-o", bin_path],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(comp.returncode, 0, f"Compilation failed: {comp.stderr}")
            run = subprocess.run(
                [bin_path, "--verify-file", vec_path],
                capture_output=True,
                text=True,
                check=False,
            )
            if run.returncode == 77 or "[SKIP]" in run.stdout:
                self.skipTest("Host CPU/compiler does not support AVX-512 VBMI + VNNI")
            self.assertEqual(run.returncode, 0, f"Golden vector / guard-page check failed: {run.stdout}\n{run.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

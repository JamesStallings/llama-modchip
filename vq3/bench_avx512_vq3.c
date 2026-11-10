// ==============================================================================
// bench_avx512_vq3.c — Zen 5 AVX-512 VBMI + VNNI Hardware Verification & Benchmark
// ==============================================================================
// Verifies and benchmarks LUT-VQ3-64 / C5-64 (24B / 64 weights = 3.00 bpw, -20% DDR5
// bus traffic) against real Fractal-64 / block_q2_0 (30B / 64 weights = 3.75 bpw).
//
// Compile:
//   gcc -O3 -march=native -fopenmp vq3/bench_avx512_vq3.c -lm -o vq3/bench_avx512_vq3
// ==============================================================================

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// 24-byte LUT-VQ3-64 / C5-64 block (3.00 bpw)
typedef struct __attribute__((packed)) {
    uint16_t d;       // FP16 super-block scale in normalized units
    uint8_t  qs[22];  // bits 0..6: 3D VQ index (0..127); bit 7: 22 scale bits; bit 6 of qs[20..21]: bits 22..23
} block_vq3_64_t;     // 24 bytes total

// 30-byte Fractal-64 (real block_q2_0 layout: d + scales[4] + qs[16] + qh[8], 3.75 bpw)
typedef struct __attribute__((packed)) {
    uint16_t d;         // FP16 super-block scale
    uint8_t  scales[4]; // 8x 4-bit 8-weight sub-scales
    uint8_t  qs[16];    // 64 x 2-bit low planes (4 per byte)
    uint8_t  qh[8];     // 64 x 1-bit high plane (8 per byte)
} block_fractal64_t;    // 30 bytes total

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VBMI__) && defined(__AVX512VNNI__)

static void *xalloc64(size_t n) {
    size_t aligned_n = (n + 63U) & ~(size_t)63U;
    if (aligned_n == 0) aligned_n = 64U;
    void *p = aligned_alloc(64U, aligned_n);
    if (!p) {
        perror("aligned_alloc");
        exit(2);
    }
    return p;
}

// Precompute 16 x int32 sums of 4 consecutive int8 activations multiplied by `offset`
// across an entire activation row (computed once per token, shared across all rows in MUL_MAT)
static void precompute_x_off4_row(const int8_t *x_q8, int32_t *x_off4_out, int n_blocks, int offset) {
    const __m512i ones_off = _mm512_set1_epi8((char)(uint8_t)offset);
    for (int b = 0; b < n_blocks; b++) {
        __m512i x = _mm512_loadu_si512((const void *)(x_q8 + (size_t)b * 64U));
        __m512i off4 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones_off, x);
        _mm512_storeu_si512((void *)(x_off4_out + (size_t)b * 16U), off4);
    }
}

// Scalar reference dot product for exact numerical verification of vec_dot_vq3_64_zen5
static float vec_dot_vq3_64_scalar_ref(const block_vq3_64_t *blocks,
                                       const int8_t *x_q8,
                                       int n_blocks,
                                       const uint8_t *T0,
                                       const uint8_t *T1,
                                       const uint8_t *T2,
                                       float g) {
    const int OFF = 128;
    float total = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        int w_s8[64];
        for (int k = 0; k < 20; k++) {
            uint8_t r = blocks[b].qs[k] & 0x7F;
            w_s8[k]      = (int)T0[r] - OFF;
            w_s8[k + 22] = (int)T1[r] - OFF;
            w_s8[k + 44] = (int)T2[r] - OFF;
        }
        for (int k = 20; k < 22; k++) {
            uint8_t r = blocks[b].qs[k] & 0x7F;
            w_s8[k]      = (int)T0[r] - OFF;
            w_s8[k + 22] = (int)T1[r] - OFF;
        }
        uint32_t s24 = 0;
        for (int k = 0; k < 22; k++) {
            if (blocks[b].qs[k] & 0x80U) s24 |= (1U << k);
        }
        s24 |= ((uint32_t)(blocks[b].qs[20] & 0x40U)) << 16;
        s24 |= ((uint32_t)(blocks[b].qs[21] & 0x40U)) << 17;

        int32_t block_sum = 0;
        for (int sb = 0; sb < 8; sb++) {
            int32_t sc = (int32_t)((s24 >> (3 * sb)) & 0x7U) + 2;
            int32_t sub_dot = 0;
            for (int j = 0; j < 8; j++) {
                int idx = sb * 8 + j;
                sub_dot += w_s8[idx] * (int32_t)x_q8[b * 64 + idx];
            }
            block_sum += sub_dot * sc;
        }
        float d = _cvtsh_ss(blocks[b].d);
        total += (float)block_sum * d;
    }
    return total * g;
}

// Scalar reference dot product for real 30-byte Fractal-64 (block_q2_0)
static float vec_dot_fractal64_scalar_ref(const block_fractal64_t *blocks,
                                          const int8_t *x_q8,
                                          int n_blocks) {
    static const int8_t cb[8] = {-14, -8, -4, -1, 1, 4, 8, 14};
    float total = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        uint32_t sc32;
        memcpy(&sc32, blocks[b].scales, 4);
        int32_t block_sum = 0;
        for (int sb = 0; sb < 8; sb++) {
            int32_t sc = (int32_t)((sc32 >> (4 * sb)) & 0xFU) + 4;
            int32_t sub_dot = 0;
            for (int j = 0; j < 8; j++) {
                int idx = sb * 8 + j;
                uint8_t low2 = (blocks[b].qs[idx >> 2] >> ((idx & 3) * 2)) & 0x3U;
                uint8_t hi1  = (blocks[b].qh[idx >> 3] >> (idx & 7)) & 0x1U;
                uint8_t q3   = low2 | (hi1 << 2);
                sub_dot += (int32_t)cb[q3] * (int32_t)x_q8[b * 64 + idx];
            }
            block_sum += sub_dot * sc;
        }
        float d = _cvtsh_ss(blocks[b].d);
        total += (float)block_sum * d;
    }
    return total;
}

// Kernel 1: LUT-VQ3-64 (24 bytes/block = 3.00 bpw)
// Uses fault-suppressing `_mm512_maskz_loadu_epi8(0xFFFFFF, ...)` so it never reads past the 24-byte block.
__attribute__((noinline))
float vec_dot_vq3_64_zen5(const block_vq3_64_t * __restrict blocks,
                          const int8_t * __restrict x_q8,
                          const int32_t * __restrict x_off4_pre,
                          int n_blocks,
                          const uint8_t *T0, const uint8_t *T1, const uint8_t *T2,
                          float g) {
    __m512i T0_lo = _mm512_loadu_si512(T0);
    __m512i T0_hi = _mm512_loadu_si512(T0 + 64);
    __m512i T1_lo = _mm512_loadu_si512(T1);
    __m512i T1_hi = _mm512_loadu_si512(T1 + 64);
    __m512i T2_lo = _mm512_loadu_si512(T2);
    __m512i T2_hi = _mm512_loadu_si512(T2 + 64);

    uint8_t rep_arr[64];
    for (int j = 0; j < 64; j++) rep_arr[j] = (uint8_t)(2 + (j % 22));
    const __m512i REP = _mm512_loadu_si512(rep_arr);

    const __mmask64 M_LOAD24 = 0x0000000000FFFFFFULL;
    const __mmask64 M0 = 0x00000000003FFFFFULL;
    const __mmask64 M1 = 0x00000FFFFFC00000ULL;
    const __mmask64 M2 = 0xFFFFF00000000000ULL;

    const __m512i s_bias = _mm512_set1_epi32(2);
    const __m512i w_1240 = _mm512_set1_epi32(0x00040201);
    const __m512i mask_01 = _mm512_set1_epi8(0x01);
    const __m512i ones_128 = _mm512_set1_epi8((char)128);

    // Pure-ZMM scale bit gather table (zero Vector->GPR round-trip!):
    // For each of the 16 int32 lanes `l` (sub-block `sb = l >> 1`), bytes 0, 1, 2 of lane `l`
    // select the 3 bytes in `raw` carrying bits 0, 1, 2 of `sb`'s 3-bit scale:
    // - sb = 0..6: Bit 7 of raw[2 + 3*sb + 0], raw[2 + 3*sb + 1], raw[2 + 3*sb + 2]
    // - sb = 7:    Bit 7 of raw[23], Bit 6 of raw[22], Bit 6 of raw[23]
    uint8_t perm_s_arr[64];
    uint16_t shift_s_arr[32];
    for (int l = 0; l < 16; l++) {
        int sb = l >> 1;
        if (sb < 7) {
            perm_s_arr[4 * l + 0] = (uint8_t)(2 + 3 * sb + 0);
            perm_s_arr[4 * l + 1] = (uint8_t)(2 + 3 * sb + 1);
            perm_s_arr[4 * l + 2] = (uint8_t)(2 + 3 * sb + 2);
            perm_s_arr[4 * l + 3] = 0;
            shift_s_arr[2 * l + 0] = 7;
            shift_s_arr[2 * l + 1] = 7;
        } else {
            // Put Bit-7 source (raw[23]) in byte 0, zero in byte 1, and Bit-6 sources (raw[22], raw[23])
            // in bytes 2 and 3 (which share the second uint16 word shifted right by 6!)
            perm_s_arr[4 * l + 0] = 23;
            perm_s_arr[4 * l + 1] = 0;
            perm_s_arr[4 * l + 2] = 22;
            perm_s_arr[4 * l + 3] = 23;
            shift_s_arr[2 * l + 0] = 7;
            shift_s_arr[2 * l + 1] = 6;
        }
    }
    const __m512i PERM_S  = _mm512_loadu_si512(perm_s_arr);
    const __m512i SHIFT_S = _mm512_loadu_si512(shift_s_arr);
    // For sb = 0..6, weights are [1, 2, 4, 0]; for sb = 7 (lanes 14, 15), bits are at bytes [0, 2, 3] -> weights [1, 0, 2, 4]
    const __m512i W_SCALE = _mm512_set_epi32(
        0x04020001, 0x04020001,
        0x00040201, 0x00040201, 0x00040201, 0x00040201,
        0x00040201, 0x00040201, 0x00040201, 0x00040201,
        0x00040201, 0x00040201, 0x00040201, 0x00040201,
        0x00040201, 0x00040201
    );
    (void)w_1240;

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    int b = 0;
    for (; b + 1 < n_blocks; b += 2) {
        __m512i raw0 = _mm512_maskz_loadu_epi8(M_LOAD24, (const void *)&blocks[b]);
        __m512i raw1 = _mm512_maskz_loadu_epi8(M_LOAD24, (const void *)&blocks[b + 1]);
        __m512i x0   = _mm512_loadu_si512((const void *)(x_q8 + (size_t)b * 64U));
        __m512i x1   = _mm512_loadu_si512((const void *)(x_q8 + (size_t)(b + 1) * 64U));

        __m512i t0 = _mm512_permutexvar_epi8(REP, raw0);
        __m512i t1 = _mm512_permutexvar_epi8(REP, raw1);
        t0 = _mm512_mask2_permutex2var_epi8(T1_lo, t0, M1, T1_hi);
        t1 = _mm512_mask2_permutex2var_epi8(T1_lo, t1, M1, T1_hi);
        t0 = _mm512_mask2_permutex2var_epi8(T2_lo, t0, M2, T2_hi);
        t1 = _mm512_mask2_permutex2var_epi8(T2_lo, t1, M2, T2_hi);
        __m512i w0 = _mm512_mask2_permutex2var_epi8(T0_lo, t0, M0, T0_hi);
        __m512i w1 = _mm512_mask2_permutex2var_epi8(T0_lo, t1, M0, T0_hi);

        __m512i dot0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w0, x0);
        __m512i dot1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w1, x1);
        __m512i off0 = x_off4_pre
            ? _mm512_loadu_si512((const void *)(x_off4_pre + (size_t)b * 16U))
            : _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones_128, x0);
        __m512i off1 = x_off4_pre
            ? _mm512_loadu_si512((const void *)(x_off4_pre + (size_t)(b + 1) * 16U))
            : _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones_128, x1);
        dot0 = _mm512_sub_epi32(dot0, off0);
        dot1 = _mm512_sub_epi32(dot1, off1);

        // Pure-ZMM scale decode via VPERMB + VPSRLVW + VPDPBUSD
        __m512i sb0 = _mm512_and_si512(_mm512_srlv_epi16(_mm512_permutexvar_epi8(PERM_S, raw0), SHIFT_S), mask_01);
        __m512i sb1 = _mm512_and_si512(_mm512_srlv_epi16(_mm512_permutexvar_epi8(PERM_S, raw1), SHIFT_S), mask_01);
        __m512i s16_0 = _mm512_dpbusd_epi32(s_bias, sb0, W_SCALE);
        __m512i s16_1 = _mm512_dpbusd_epi32(s_bias, sb1, W_SCALE);

        float d0 = _cvtsh_ss(blocks[b].d);
        float d1 = _cvtsh_ss(blocks[b + 1].d);
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_mullo_epi32(dot0, s16_0)), _mm512_set1_ps(d0), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_mullo_epi32(dot1, s16_1)), _mm512_set1_ps(d1), acc1);
    }

    for (; b < n_blocks; b++) {
        __m512i raw = _mm512_maskz_loadu_epi8(M_LOAD24, (const void *)&blocks[b]);
        __m512i x   = _mm512_loadu_si512((const void *)(x_q8 + (size_t)b * 64U));

        __m512i t = _mm512_permutexvar_epi8(REP, raw);
        t = _mm512_mask2_permutex2var_epi8(T1_lo, t, M1, T1_hi);
        t = _mm512_mask2_permutex2var_epi8(T2_lo, t, M2, T2_hi);
        __m512i w_u8 = _mm512_mask2_permutex2var_epi8(T0_lo, t, M0, T0_hi);

        __m512i dot16 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_u8, x);
        __m512i x_off4 = x_off4_pre
            ? _mm512_loadu_si512((const void *)(x_off4_pre + (size_t)b * 16U))
            : _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones_128, x);
        dot16 = _mm512_sub_epi32(dot16, x_off4);

        __m512i sb_bits = _mm512_and_si512(_mm512_srlv_epi16(_mm512_permutexvar_epi8(PERM_S, raw), SHIFT_S), mask_01);
        __m512i s16 = _mm512_dpbusd_epi32(s_bias, sb_bits, W_SCALE);

        float d = _cvtsh_ss(blocks[b].d);
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_mullo_epi32(dot16, s16)), _mm512_set1_ps(d), acc0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) * g;
}

// Kernel 2: True Fractal-64 / block_q2_0 Baseline (30 bytes/block = 3.75 bpw)
// Decodes the exact `d (2B) + scales (4B) + qs (16B, 2-bit low) + qh (8B, 1-bit high via vprorvq)` layout.
__attribute__((noinline))
float vec_dot_fractal64_zen5(const block_fractal64_t * __restrict blocks,
                             const int8_t * __restrict x_q8,
                             const int32_t * __restrict x_off14_pre,
                             int n_blocks) {
    const __m512i lut = _mm512_set1_epi64(0x1C16120F0D0A0600ULL);
    const __m512i mask2 = _mm512_set1_epi8(0x03);
    const __m512i bit2  = _mm512_set1_epi8(0x04);
    const __m512i s_mask = _mm512_set1_epi32(0x0F);
    const __m512i s_bias = _mm512_set1_epi32(4);
    const __m512i shuf_s = _mm512_set_epi32(7,7,6,6, 5,5,4,4, 3,3,2,2, 1,1,0,0);
    const __m512i ones_14 = _mm512_set1_epi8(14);
    const __mmask64 M_LOAD30 = 0x000000003FFFFFFFULL;

    // Each byte qs[m] (at raw offset 6 + m, m=0..15) holds 4 2-bit values for weights 4m..4m+3
    uint8_t p_qs[64];
    for (int i = 0; i < 64; i++) p_qs[i] = (uint8_t)(6 + (i >> 2));
    const __m512i perm_qs = _mm512_loadu_si512(p_qs);

    // Variable 16-bit right shifts for lanes i%4 == 0,1,2,3 -> shift by 0,2,4,6
    uint16_t sh_qs[32];
    for (int i = 0; i < 32; i++) sh_qs[i] = (uint16_t)(((i * 2) & 3) * 2);
    const __m512i shift_even = _mm512_loadu_si512(sh_qs);
    for (int i = 0; i < 32; i++) sh_qs[i] = (uint16_t)((((i * 2 + 1) & 3) * 2) + 8);
    const __m512i shift_odd  = _mm512_loadu_si512(sh_qs);

    // Zen 5 `_mm512_rorv_epi64` shifts for extracting 64 high bits from qh (8 bytes)
    const __m512i ror_qh = _mm512_set_epi64(54, 46, 38, 30, 22, 14, 6, 62);
    const __m512i mask_qh_bit = _mm512_set1_epi64(0x0402010080402010ULL);
    uint8_t p_qh[64];
    for (int i = 0; i < 64; i++) p_qh[i] = (uint8_t)(22 + (i >> 3));
    const __m512i perm_qh = _mm512_loadu_si512(p_qh);
    uint8_t bit_mask_arr[64];
    for (int i = 0; i < 64; i++) bit_mask_arr[i] = (uint8_t)(1U << (i & 7));
    const __m512i qh_bit_mask = _mm512_loadu_si512(bit_mask_arr);
    (void)ror_qh; (void)mask_qh_bit;

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    int b = 0;
    for (; b + 1 < n_blocks; b += 2) {
        __m512i raw0 = _mm512_maskz_loadu_epi8(M_LOAD30, (const void *)&blocks[b]);
        __m512i raw1 = _mm512_maskz_loadu_epi8(M_LOAD30, (const void *)&blocks[b + 1]);
        __m512i x0   = _mm512_loadu_si512((const void *)(x_q8 + (size_t)b * 64U));
        __m512i x1   = _mm512_loadu_si512((const void *)(x_q8 + (size_t)(b + 1) * 64U));

        __m512i q_rep0  = _mm512_permutexvar_epi8(perm_qs, raw0);
        __m512i q_rep1  = _mm512_permutexvar_epi8(perm_qs, raw1);
        __m512i q_even0 = _mm512_srlv_epi16(q_rep0, shift_even);
        __m512i q_even1 = _mm512_srlv_epi16(q_rep1, shift_even);
        __m512i q_odd0  = _mm512_srlv_epi16(q_rep0, shift_odd);
        __m512i q_odd1  = _mm512_srlv_epi16(q_rep1, shift_odd);
        __m512i low2_0  = _mm512_and_si512(_mm512_mask_mov_epi8(q_even0, 0xAAAAAAAAAAAAAAAAULL, _mm512_slli_epi16(q_odd0, 8)), mask2);
        __m512i low2_1  = _mm512_and_si512(_mm512_mask_mov_epi8(q_even1, 0xAAAAAAAAAAAAAAAAULL, _mm512_slli_epi16(q_odd1, 8)), mask2);

        __m512i h_rep0  = _mm512_permutexvar_epi8(perm_qh, raw0);
        __m512i h_rep1  = _mm512_permutexvar_epi8(perm_qh, raw1);
        __mmask64 h_m0  = _mm512_test_epi8_mask(h_rep0, qh_bit_mask);
        __mmask64 h_m1  = _mm512_test_epi8_mask(h_rep1, qh_bit_mask);
        __m512i idx0    = _mm512_mask_add_epi8(low2_0, h_m0, low2_0, bit2);
        __m512i idx1    = _mm512_mask_add_epi8(low2_1, h_m1, low2_1, bit2);

        __m512i w_u8_0  = _mm512_shuffle_epi8(lut, idx0);
        __m512i w_u8_1  = _mm512_shuffle_epi8(lut, idx1);
        __m512i dot0    = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_u8_0, x0);
        __m512i dot1    = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_u8_1, x1);
        __m512i off0 = x_off14_pre
            ? _mm512_loadu_si512((const void *)(x_off14_pre + (size_t)b * 16U))
            : _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones_14, x0);
        __m512i off1 = x_off14_pre
            ? _mm512_loadu_si512((const void *)(x_off14_pre + (size_t)(b + 1) * 16U))
            : _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones_14, x1);
        dot0 = _mm512_sub_epi32(dot0, off0);
        dot1 = _mm512_sub_epi32(dot1, off1);

        uint32_t sc0, sc1;
        memcpy(&sc0, blocks[b].scales, 4);
        memcpy(&sc1, blocks[b + 1].scales, 4);
        __m256i s8_0 = _mm256_srlv_epi32(_mm256_set1_epi32((int)sc0), _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28));
        __m256i s8_1 = _mm256_srlv_epi32(_mm256_set1_epi32((int)sc1), _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28));
        __m512i s16_0 = _mm512_add_epi32(_mm512_and_epi32(_mm512_permutexvar_epi32(shuf_s, _mm512_castsi256_si512(s8_0)), s_mask), s_bias);
        __m512i s16_1 = _mm512_add_epi32(_mm512_and_epi32(_mm512_permutexvar_epi32(shuf_s, _mm512_castsi256_si512(s8_1)), s_mask), s_bias);

        float d0 = _cvtsh_ss(blocks[b].d);
        float d1 = _cvtsh_ss(blocks[b + 1].d);
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_mullo_epi32(dot0, s16_0)), _mm512_set1_ps(d0), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_mullo_epi32(dot1, s16_1)), _mm512_set1_ps(d1), acc1);
    }

    for (; b < n_blocks; b++) {
        __m512i raw = _mm512_maskz_loadu_epi8(M_LOAD30, (const void *)&blocks[b]);
        __m512i x   = _mm512_loadu_si512((const void *)(x_q8 + (size_t)b * 64U));

        __m512i q_rep  = _mm512_permutexvar_epi8(perm_qs, raw);
        __m512i q_even = _mm512_srlv_epi16(q_rep, shift_even);
        __m512i q_odd  = _mm512_srlv_epi16(q_rep, shift_odd);
        __m512i low2   = _mm512_and_si512(_mm512_mask_mov_epi8(q_even, 0xAAAAAAAAAAAAAAAAULL, _mm512_slli_epi16(q_odd, 8)), mask2);

        __m512i h_rep  = _mm512_permutexvar_epi8(perm_qh, raw);
        __mmask64 h_m  = _mm512_test_epi8_mask(h_rep, qh_bit_mask);
        __m512i idx    = _mm512_mask_add_epi8(low2, h_m, low2, bit2);

        __m512i w_u8   = _mm512_shuffle_epi8(lut, idx);
        __m512i dot16  = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_u8, x);
        __m512i x_off4 = x_off14_pre
            ? _mm512_loadu_si512((const void *)(x_off14_pre + (size_t)b * 16U))
            : _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones_14, x);
        dot16 = _mm512_sub_epi32(dot16, x_off4);

        uint32_t sc32;
        memcpy(&sc32, blocks[b].scales, 4);
        __m256i s8 = _mm256_srlv_epi32(
            _mm256_set1_epi32((int)sc32),
            _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28)
        );
        __m512i s16 = _mm512_permutexvar_epi32(shuf_s, _mm512_castsi256_si512(s8));
        s16 = _mm512_add_epi32(_mm512_and_epi32(s16, s_mask), s_bias);

        __m512i scaled = _mm512_mullo_epi32(dot16, s16);
        uint16_t d_u16 = (uint16_t)_mm_extract_epi16(_mm512_castsi512_si128(raw), 0);
        float d = _cvtsh_ss(d_u16);
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(scaled), _mm512_set1_ps(d), acc0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

static int verify_guard_page_safety(const uint8_t *T0, const uint8_t *T1, const uint8_t *T2) {
    long page_sz = sysconf(_SC_PAGESIZE);
    if (page_sz <= 0) page_sz = 4096;
    uint8_t *mem = mmap(NULL, (size_t)page_sz * 2U, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return -1;
    if (mprotect(mem + page_sz, (size_t)page_sz, PROT_NONE) != 0) {
        munmap(mem, (size_t)page_sz * 2U);
        return -1;
    }

    int8_t *x_q8 = xalloc64(4 * 64);
    for (int i = 0; i < 256; i++) x_q8[i] = (int8_t)((i % 31) - 15);

    // Place 4 LUT-VQ3-64 blocks (96 bytes) flush against the PROT_NONE page boundary
    block_vq3_64_t *vq_tail = (block_vq3_64_t *)(mem + page_sz - 4 * sizeof(block_vq3_64_t));
    for (int b = 0; b < 4; b++) {
        vq_tail[b].d = 0x3800;
        for (int k = 0; k < 22; k++) vq_tail[b].qs[k] = (uint8_t)((b * 17 + k * 13) & 0xFF);
    }
    float vq_avx = vec_dot_vq3_64_zen5(vq_tail, x_q8, NULL, 4, T0, T1, T2, 0.25f);
    float vq_ref = vec_dot_vq3_64_scalar_ref(vq_tail, x_q8, 4, T0, T1, T2, 0.25f);

    // Place 4 Fractal-64 blocks (120 bytes) flush against the PROT_NONE page boundary
    block_fractal64_t *f64_tail = (block_fractal64_t *)(mem + page_sz - 4 * sizeof(block_fractal64_t));
    for (int b = 0; b < 4; b++) {
        f64_tail[b].d = 0x3800;
        for (int k = 0; k < 4; k++)  f64_tail[b].scales[k] = (uint8_t)(b * 3 + k * 5);
        for (int k = 0; k < 16; k++) f64_tail[b].qs[k] = (uint8_t)(b * 11 + k * 7);
        for (int k = 0; k < 8; k++)  f64_tail[b].qh[k] = (uint8_t)(b * 19 + k * 23);
    }
    float f64_avx = vec_dot_fractal64_zen5(f64_tail, x_q8, NULL, 4);
    float f64_ref = vec_dot_fractal64_scalar_ref(f64_tail, x_q8, 4);

    free(x_q8);
    munmap(mem, (size_t)page_sz * 2U);

    if (fabsf(vq_avx - vq_ref) > 1e-4f || fabsf(f64_avx - f64_ref) > 1e-4f) {
        printf(" [FAIL] Guard-page kernel mismatch: vq=(%.4f vs %.4f), f64=(%.4f vs %.4f)\n",
               vq_avx, vq_ref, f64_avx, f64_ref);
        return -1;
    }
    return 0;
}

static int verify_zen5_hardware_vq3(void) {
    uint8_t T0[128], T1[128], T2[128];
    const int OFF = 128;

    for (int r = 0; r < 128; r++) {
        int c0 = ((r * 19 + 7) % 241) - 120;
        int c1 = ((r * 37 + 13) % 241) - 120;
        int c2 = ((r * 53 + 29) % 241) - 120;
        T0[r] = (uint8_t)(c0 + OFF);
        T1[r] = (uint8_t)(c1 + OFF);
        T2[r] = (uint8_t)(c2 + OFF);
    }

    const int n_test = 64;
    block_vq3_64_t *blks = xalloc64(n_test * sizeof(block_vq3_64_t));
    block_fractal64_t *fblks = xalloc64(n_test * sizeof(block_fractal64_t));
    int8_t *x_q8 = xalloc64(n_test * 64);

    for (int b = 0; b < n_test; b++) {
        blks[b].d = 0x3800; // FP16 0.5
        for (int k = 0; k < 22; k++) {
            blks[b].qs[k] = (uint8_t)((b * 13 + k * 29 + 5) & 0xFF);
        }
        fblks[b].d = 0x3800;
        for (int k = 0; k < 4; k++)  fblks[b].scales[k] = (uint8_t)((b * 7 + k * 13) & 0xFF);
        for (int k = 0; k < 16; k++) fblks[b].qs[k] = (uint8_t)((b * 17 + k * 31) & 0xFF);
        for (int k = 0; k < 8; k++)  fblks[b].qh[k] = (uint8_t)((b * 23 + k * 41) & 0xFF);
        for (int j = 0; j < 64; j++) {
            x_q8[b * 64 + j] = (int8_t)(((b * 7 + j * 11) % 200) - 100);
        }
    }

    float g = 0.0125f;
    float dot_avx = vec_dot_vq3_64_zen5(blks, x_q8, NULL, n_test, T0, T1, T2, g);
    float dot_ref = vec_dot_vq3_64_scalar_ref(blks, x_q8, n_test, T0, T1, T2, g);
    float f64_avx = vec_dot_fractal64_zen5(fblks, x_q8, NULL, n_test);
    float f64_ref = vec_dot_fractal64_scalar_ref(fblks, x_q8, n_test);
    free(blks);
    free(fblks);
    free(x_q8);

    float rel_err_vq = fabsf(dot_avx - dot_ref) / (fabsf(dot_ref) + 1e-6f);
    float rel_err_f64 = fabsf(f64_avx - f64_ref) / (fabsf(f64_ref) + 1e-6f);
    if (rel_err_vq > 1e-5f || rel_err_f64 > 1e-5f) {
        printf(" [FAIL] Kernel mismatch: vq_err=%.2e, f64_err=%.2e\n", rel_err_vq, rel_err_f64);
        return -1;
    }

    return verify_guard_page_safety(T0, T1, T2);
}

static int verify_from_python_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return 1;
    }
    int32_t n_blocks = 0;
    float g = 0.0f;
    float expected_dot = 0.0f;
    uint8_t T0[128], T1[128], T2[128];

    if (fread(&n_blocks, sizeof(int32_t), 1, f) != 1 || n_blocks <= 0 || n_blocks > 1000000) {
        fclose(f);
        return 1;
    }
    if (fread(&g, sizeof(float), 1, f) != 1 ||
        fread(&expected_dot, sizeof(float), 1, f) != 1 ||
        fread(T0, 1, 128, f) != 128 ||
        fread(T1, 1, 128, f) != 128 ||
        fread(T2, 1, 128, f) != 128) {
        fclose(f);
        return 1;
    }
    block_vq3_64_t *blks = xalloc64((size_t)n_blocks * sizeof(block_vq3_64_t));
    int8_t *x_q8 = xalloc64((size_t)n_blocks * 64U);
    if (fread(blks, sizeof(block_vq3_64_t), (size_t)n_blocks, f) != (size_t)n_blocks ||
        fread(x_q8, 1, (size_t)n_blocks * 64U, f) != (size_t)n_blocks * 64U) {
        free(blks);
        free(x_q8);
        fclose(f);
        return 1;
    }
    fclose(f);

    float dot_avx = vec_dot_vq3_64_zen5(blks, x_q8, NULL, n_blocks, T0, T1, T2, g);
    float dot_ref = vec_dot_vq3_64_scalar_ref(blks, x_q8, n_blocks, T0, T1, T2, g);
    free(blks);
    free(x_q8);

    float rel_py  = fabsf(dot_avx - expected_dot) / (fabsf(expected_dot) + 1e-6f);
    float rel_ref = fabsf(dot_avx - dot_ref) / (fabsf(dot_ref) + 1e-6f);
    if (rel_py > 1e-5f || rel_ref > 1e-5f) {
        fprintf(stderr, " [FAIL] Golden vector mismatch: avx=%.6f, py=%.6f, ref=%.6f\n",
                dot_avx, expected_dot, dot_ref);
        return 1;
    }
    printf(" [PASS] Python -> C AVX-512 VBMI golden vector verified (rel_err=%.2e)!\n", rel_py);
    return 0;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    if (argc > 2 && strcmp(argv[1], "--verify-file") == 0) {
        if (verify_zen5_hardware_vq3() != 0) return 1;
        return verify_from_python_file(argv[2]);
    }
    int quick_mode = (argc > 1 && strcmp(argv[1], "--quick") == 0);
    printf("==============================================================================\n");
    printf(" PHASE 2: ZEN 5 AVX-512 VBMI + VNNI SILICON BENCHMARK\n");
    printf("==============================================================================\n");

    if (verify_zen5_hardware_vq3() != 0) return 1;
    printf(" [PASS] Real AVX-512 VBMI+VNNI silicon verified: VPERMI2B ignores bit 7, guard-page safe & matches scalar ref!\n");

    if (quick_mode) return 0;

    uint8_t T0[128], T1[128], T2[128];
    for (int i = 0; i < 128; i++) {
        T0[i] = (uint8_t)((i * 3) & 0xFF);
        T1[i] = (uint8_t)((i * 5 + 17) & 0xFF);
        T2[i] = (uint8_t)((i * 7 + 31) & 0xFF);
    }

    const int n_l1_blocks = 1024;
    const int l1_reps = 100000;
    block_fractal64_t *f64_l1 = xalloc64(n_l1_blocks * sizeof(block_fractal64_t));
    block_vq3_64_t    *vq3_l1 = xalloc64(n_l1_blocks * sizeof(block_vq3_64_t));
    int8_t            *x_l1   = xalloc64(n_l1_blocks * 64);
    int32_t           *off14_l1 = xalloc64(n_l1_blocks * 16 * sizeof(int32_t));
    int32_t           *off128_l1 = xalloc64(n_l1_blocks * 16 * sizeof(int32_t));

    for (int i = 0; i < n_l1_blocks * 64; i++) x_l1[i] = (int8_t)((i * 13 + 7) % 127 - 63);
    for (int b = 0; b < n_l1_blocks; b++) {
        f64_l1[b].d = 0x3800;
        for (int k = 0; k < 4; k++)  f64_l1[b].scales[k] = (uint8_t)(b + k * 17);
        for (int k = 0; k < 16; k++) f64_l1[b].qs[k] = (uint8_t)(b * 3 + k * 29);
        for (int k = 0; k < 8; k++)  f64_l1[b].qh[k] = (uint8_t)(b * 7 + k * 11);
        vq3_l1[b].d = 0x3800;
        for (int k = 0; k < 22; k++) vq3_l1[b].qs[k] = (uint8_t)(b * 5 + k * 23);
    }
    precompute_x_off4_row(x_l1, off14_l1, n_l1_blocks, 14);
    precompute_x_off4_row(x_l1, off128_l1, n_l1_blocks, 128);

    volatile float sink = 0.0f;
    struct timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < l1_reps; r++) {
        sink += vec_dot_fractal64_zen5(f64_l1, x_l1, off14_l1, n_l1_blocks);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_f64_l1 = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double ns_per_blk_f64 = (dt_f64_l1 * 1e9) / ((double)n_l1_blocks * l1_reps);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < l1_reps; r++) {
        sink += vec_dot_vq3_64_zen5(vq3_l1, x_l1, off128_l1, n_l1_blocks, T0, T1, T2, 0.01f);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_vq3_l1 = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double ns_per_blk_vq3 = (dt_vq3_l1 * 1e9) / ((double)n_l1_blocks * l1_reps);

    printf(" • L1/L2 Single-Core Decode+VNNI Speed (64 weights/block):\n");
    printf("   - Fractal-64 (30B, 3.75 bpw): %.2f ns/block (%.1f GWeights/s single-core)\n",
           ns_per_blk_f64, 64.0 / ns_per_blk_f64);
    printf("   - LUT-VQ3-64 (24B, 3.00 bpw): %.2f ns/block (%.1f GWeights/s single-core)\n",
           ns_per_blk_vq3, 64.0 / ns_per_blk_vq3);

    const long n_ram_blocks = 16777216L;
    const int n_rows = 4096;
    const int blks_per_row = (int)(n_ram_blocks / n_rows);

    block_fractal64_t *f64_ram = xalloc64((size_t)n_ram_blocks * sizeof(block_fractal64_t));
    block_vq3_64_t    *vq3_ram = xalloc64((size_t)n_ram_blocks * sizeof(block_vq3_64_t));
    int8_t            *x_ram   = xalloc64((size_t)blks_per_row * 64U);
    int32_t           *off14_ram  = xalloc64((size_t)blks_per_row * 16U * sizeof(int32_t));
    int32_t           *off128_ram = xalloc64((size_t)blks_per_row * 16U * sizeof(int32_t));

    memset(f64_ram, 0x35, (size_t)n_ram_blocks * sizeof(block_fractal64_t));
    memset(vq3_ram, 0x35, (size_t)n_ram_blocks * sizeof(block_vq3_64_t));
    for (int i = 0; i < blks_per_row * 64; i++) x_ram[i] = (int8_t)((i * 7 + 3) % 63 - 31);
    precompute_x_off4_row(x_ram, off14_ram, blks_per_row, 14);
    precompute_x_off4_row(x_ram, off128_ram, blks_per_row, 128);

    const int ram_iters = 41;
    double samples_f64[41], samples_vq3[41];

    // Interleaved trials to eliminate thermal/boost drift between Fractal-64 and LUT-VQ3-64
    for (int it = 0; it < ram_iters; it++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        #pragma omp parallel for schedule(static) reduction(+:sink)
        for (int row = 0; row < n_rows; row++) {
            sink += vec_dot_fractal64_zen5(f64_ram + (long)row * blks_per_row, x_ram, off14_ram, blks_per_row);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        samples_f64[it] = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        #pragma omp parallel for schedule(static) reduction(+:sink)
        for (int row = 0; row < n_rows; row++) {
            sink += vec_dot_vq3_64_zen5(vq3_ram + (long)row * blks_per_row, x_ram, off128_ram, blks_per_row, T0, T1, T2, 0.01f);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        samples_vq3[it] = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    }

    qsort(samples_f64, ram_iters, sizeof(double), cmp_double);
    qsort(samples_vq3, ram_iters, sizeof(double), cmp_double);
    double dt_f64_ram = samples_f64[ram_iters / 2];
    double dt_vq3_ram = samples_vq3[ram_iters / 2];

    double bw_f64 = (n_ram_blocks * 30.0) / dt_f64_ram / 1e9;
    double gw_f64 = (n_ram_blocks * 64.0) / dt_f64_ram / 1e9;
    double bw_vq3 = (n_ram_blocks * 24.0) / dt_vq3_ram / 1e9;
    double gw_vq3 = (n_ram_blocks * 64.0) / dt_vq3_ram / 1e9;

    printf(" • Multi-Core DDR5 RAM Streaming Median of %d Interleaved Passes (1.07B weights/pass):\n", ram_iters);
    printf("   - Fractal-64 (30B, 3.75 bpw): %.2f ms/pass | %.2f GB/s DDR5 | %.2f Billion weights/s\n",
           dt_f64_ram * 1e3, bw_f64, gw_f64);
    printf("   - LUT-VQ3-64 (24B, 3.00 bpw): %.2f ms/pass | %.2f GB/s DDR5 | %.2f Billion weights/s (%+.1f%% wall-clock speedup!)\n",
           dt_vq3_ram * 1e3, bw_vq3, gw_vq3, (gw_vq3 / gw_f64 - 1.0) * 100.0);

    free(f64_l1); free(vq3_l1); free(x_l1); free(off14_l1); free(off128_l1);
    free(f64_ram); free(vq3_ram); free(x_ram); free(off14_ram); free(off128_ram);
    return (sink == 123456789.0f) ? 1 : 0;
}

#else
int main(void) {
    printf(" [SKIP] Host compiler/CPU does not enable AVX-512 VBMI + VNNI.\n");
    return 77;
}
#endif

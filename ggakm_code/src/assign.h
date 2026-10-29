#ifndef AKM_MYKM_ASSIGN_H
#define AKM_MYKM_ASSIGN_H

#include <omp.h>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <cstring>

#include "kernels.h"
#include "workspace.h"

namespace mykm {

constexpr int X_ROW_L3   = 512;
constexpr int X_ROW_L2   = 128;
constexpr int CEN_ROW_L2 = 512;

static inline void assign_l1_masked(const float* X,
                             const float* Cen, const float* cnorm,
                             const uint8_t* maskp, int cen_rows, int c_off, int dim, int mstride,
                             float* best, int* bestj) {
    __m512  best_vec  = _mm512_loadu_ps(best);
    __m512i bestj_vec = _mm512_loadu_si512(reinterpret_cast<const void*>(bestj));
    const __m512 twos = _mm512_set1_ps(2.0f);
    for (int j0 = 0; j0 < cen_rows; j0 += 64) {
        uint64_t rb[16], grp[4] = {};
        for (int t = 0; t < 16; t++) {
            std::memcpy(&rb[t], maskp + (size_t)t * mstride + (size_t)j0 / 8, 8);
            grp[t / 4] |= rb[t];
        }
        uint64_t active = grp[0] | grp[1] | grp[2] | grp[3];
        while (active) {
            const int off = __builtin_ctzll(active); active &= (active - 1);
            const int j = j0 + off;
            const float* cjp = Cen + (size_t)j * dim;
            float dots[16] = {};
            __mmask16 amask = 0;
            for (int g = 0; g < 4; g++) {
                if ((grp[g] >> off) & 1) {            
                    const int b = g * 4;
                    micro_4x1_dots(X+(size_t)b*dim, X+(size_t)(b+1)*dim, X+(size_t)(b+2)*dim, X+(size_t)(b+3)*dim,
                                   cjp, dim, dots + b);
                }
                amask |= (__mmask16)(((rb[g*4]   >> off) & 1) << (g*4))
                      |  (__mmask16)(((rb[g*4+1] >> off) & 1) << (g*4+1))
                      |  (__mmask16)(((rb[g*4+2] >> off) & 1) << (g*4+2))
                      |  (__mmask16)(((rb[g*4+3] >> off) & 1) << (g*4+3));
            }
            __m512  dist = _mm512_fnmadd_ps(_mm512_loadu_ps(dots), twos, _mm512_set1_ps(cnorm[j]));
            __mmask16 imp = _mm512_cmp_ps_mask(dist, best_vec, _CMP_LT_OS) & amask;
            best_vec  = _mm512_mask_blend_ps(imp, best_vec, dist);
            bestj_vec = _mm512_mask_blend_epi32(imp, bestj_vec, _mm512_set1_epi32(c_off + j));
        }
    }
    _mm512_storeu_ps(best, best_vec);
    _mm512_storeu_si512(reinterpret_cast<void*>(bestj), bestj_vec);
}

static inline void assign_l1_dense(const float* X,
                             const float* Cen, const float* cnorm,
                             int cen_rows, int c_off, int dim,
                             float* best, int* bestj) {
    __m512  best_vec  = _mm512_loadu_ps(best);
    __m512i bestj_vec = _mm512_loadu_si512(reinterpret_cast<const void*>(bestj));
    const __m512 twos = _mm512_set1_ps(2.0f);
    for (int j = 0; j < cen_rows; j++) {
        float dots[16];
        micro_16x1_dots(X+(size_t)0*dim, X+(size_t)1*dim, X+(size_t)2*dim, X+(size_t)3*dim,
                        X+(size_t)4*dim, X+(size_t)5*dim, X+(size_t)6*dim, X+(size_t)7*dim,
                        X+(size_t)8*dim, X+(size_t)9*dim, X+(size_t)10*dim, X+(size_t)11*dim,
                        X+(size_t)12*dim, X+(size_t)13*dim, X+(size_t)14*dim, X+(size_t)15*dim,
                        Cen + (size_t)j * dim, dim, dots);
        __m512  dist = _mm512_fnmadd_ps(_mm512_loadu_ps(dots), twos, _mm512_set1_ps(cnorm[j]));
        __mmask16 imp = _mm512_cmp_ps_mask(dist, best_vec, _CMP_LT_OS);   
        best_vec  = _mm512_mask_blend_ps(imp, best_vec, dist);
        bestj_vec = _mm512_mask_blend_epi32(imp, bestj_vec, _mm512_set1_epi32(c_off + j));
    }
    _mm512_storeu_ps(best, best_vec);
    _mm512_storeu_si512(reinterpret_cast<void*>(bestj), bestj_vec);
}

static inline void assign_l1_tail(const float* X, int n_rows,
                             const float* Cen, const float* cnorm,
                             const uint8_t* maskp, bool use_mask,
                             int cen_rows, int c_off, int dim, int mstride,
                             float* best, int* bestj) {
    if (use_mask) {
        for (int j0 = 0; j0 < cen_rows; j0 += 64) {
            uint64_t rb[16] = {}, active = 0;
            for (int t = 0; t < n_rows; t++) {
                std::memcpy(&rb[t], maskp + (size_t)t * mstride + (size_t)j0 / 8, 8);
                active |= rb[t];
            }
            while (active) {
                const int off = __builtin_ctzll(active); active &= (active - 1);
                const int j = j0 + off;
                const float cn = cnorm[j];
                for (int t = 0; t < n_rows; t++) {
                    if ((rb[t] >> off) & 1) {
                        float dot;
                        micro_1x1_dots(X + (size_t)t * dim, Cen + (size_t)j * dim, dim, &dot);
                        float d = -2.0f * dot + cn;
                        if (d < best[t]) { best[t] = d; bestj[t] = c_off + j; }
                    }
                }
            }
        }
    } else {
        for (int j = 0; j < cen_rows; j++) {
            const float* cjp = Cen + (size_t)j * dim;
            const float cn = cnorm[j];
            for (int t = 0; t < n_rows; t++) {
                float dot;
                micro_1x1_dots(X + (size_t)t * dim, cjp, dim, &dot);
                float d = cn - 2.0f * dot;
                if (d < best[t]) { best[t] = d; bestj[t] = c_off + j; }
            }
        }
    }
}

static inline void assign_l2(const float* Xp, int x_rows, const float* Cen, const float* cnorm,
                             const uint8_t* maskp, bool use_mask,
                             int cen_rows, int c_off, int dim, int mstride,
                             float* best, int* bestj) {
    int t_off = 0;
    for (; t_off + 16 <= x_rows; t_off += 16) {
        if (use_mask)
            assign_l1_masked(Xp + (size_t)t_off * dim, Cen, cnorm,
                             maskp + (size_t)t_off * mstride, cen_rows, c_off, dim, mstride,
                             best + t_off, bestj + t_off);
        else
            assign_l1_dense(Xp + (size_t)t_off * dim, Cen, cnorm,
                            cen_rows, c_off, dim,
                            best + t_off, bestj + t_off);
    }
    if (t_off < x_rows) {
        const int n_rows = x_rows - t_off;
        assign_l1_tail(Xp + (size_t)t_off * dim, n_rows, Cen, cnorm,
                       maskp + (size_t)t_off * mstride, use_mask, cen_rows, c_off, dim, mstride,
                       best + t_off, bestj + t_off);
    }
}

static void assign_l3(Workspace& ws, int x_off, int x_rows, int tid, bool use_mask) {
    const int k = ws.params.k, dim = ws.params.dim, mstride = ws.params.mstride;
    const float*  X     = ws.data.X;
    const float*  Cen   = ws.data.Cen;
    const float*  cnorm = ws.data.c_norm.data();
    const uint8_t* mask = ws.accel.mask;

    float best[X_ROW_L3];
    int   bestj[X_ROW_L3];
    for (int r = 0; r < x_rows; r++) { best[r] = std::numeric_limits<float>::infinity(); bestj[r] = 0; }

    for (int c_off = 0; c_off < k; c_off += CEN_ROW_L2) {
        const int cen_rows = std::min(CEN_ROW_L2, k - c_off);
        for (int p_off = 0; p_off < x_rows; p_off += X_ROW_L2) {
            const int p_rows = std::min(X_ROW_L2, x_rows - p_off);
            assign_l2(X + (size_t)(x_off + p_off) * dim, p_rows,
                      Cen + (size_t)c_off * dim, cnorm + c_off,
                      mask + (size_t)(x_off + p_off) * mstride + (size_t)c_off / 8, use_mask,
                      cen_rows, c_off, dim, mstride, best + p_off, bestj + p_off);
        }
    }

    int*         labels  = ws.data.labels.data();
    float*       wc_t    = ws.data.wc_threads.data()  + (size_t)tid * k;
    float*       Cen_t   = ws.data.Cen_threads.data() + (size_t)tid * k * dim;
    const float* x_norm  = ws.data.x_norm.data();
    float*       dist_sq = ws.data.dist_sq.data();
    for (int r = 0; r < x_rows; r++) {
        const int i = x_off + r;
        const int g = bestj[r];
        labels[i]  = g;
        wc_t[g]   += 1.0f;
        dist_sq[i] = best[r] + x_norm[i];
        const float* xi = X + (size_t)i * dim;
        float* ct = Cen_t + (size_t)g * dim;
        for (int d = 0; d < dim; d++) ct[d] += xi[d];
    }
}

static void e_step(Workspace& ws, bool use_mask) {
    #pragma omp parallel for num_threads(ws.params.n_threads) schedule(auto)
    for (int x_off = 0; x_off < ws.params.n; x_off += X_ROW_L3) {
        const int x_rows = std::min(X_ROW_L3, ws.params.n - x_off);
        assign_l3(ws, x_off, x_rows, omp_get_thread_num(), use_mask);
    }
}

static void e_step_list(Workspace& ws) {
    const int n = ws.params.n, k = ws.params.k, dim = ws.params.dim, nt = ws.params.n_threads, m = ws.params.m;
    const int cap = m + 1;
    const float* X      = ws.data.X;
    const float* Cen    = ws.data.Cen;
    const float* cnorm  = ws.data.c_norm.data();
    const float* x_norm = ws.data.x_norm.data();
    const int*   cand_idx = ws.accel.cand_idx.data();
    const int*   cand_cnt = ws.accel.cand_cnt.data();
    int*   labels  = ws.data.labels.data();
    float* dist_sq = ws.data.dist_sq.data();
    #pragma omp parallel for num_threads(nt) schedule(auto)
    for (int i = 0; i < n; i++) {
        const int tid = omp_get_thread_num();
        float* wc_t  = ws.data.wc_threads.data()  + (size_t)tid * k;
        float* Cen_t = ws.data.Cen_threads.data() + (size_t)tid * k * dim;
        const int* cand = cand_idx + (size_t)i * cap;
        const int cnt = cand_cnt[i];
        const float* xi = X + (size_t)i * dim;
        float best = std::numeric_limits<float>::infinity();
        int   bj   = labels[i];
        for (int t = 0; t < cnt; t++) {
            const int j = cand[t];
            float dot = dot_product_512(xi, Cen + (size_t)j * dim, dim);
            float d = cnorm[j] - 2.0f * dot;
            if (d < best) { best = d; bj = j; }
        }
        labels[i]   = bj;
        wc_t[bj]   += 1.0f;
        dist_sq[i]  = best + x_norm[i];
        float* ct = Cen_t + (size_t)bj * dim;
        for (int d = 0; d < dim; d++) ct[d] += xi[d];
    }
}

} 
#endif  

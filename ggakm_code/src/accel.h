#ifndef AKM_MYKM_ACCEL_H
#define AKM_MYKM_ACCEL_H

#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>   

#include "kernels.h"    
#include "workspace.h"

namespace mykm {

static void build_knn(Workspace& ws) {
    if (ws.params.m <= 0) return;
    const int k = ws.params.k, m = ws.params.m, nt = ws.params.n_threads, dim = ws.params.dim;
    const float* c_norm = ws.data.c_norm.data();
    int*   knn = ws.accel.knn_idx.data();
    float* ecs = ws.accel.edge_cos.data();
    float* esi = ws.accel.edge_sin.data();
    float* edt = ws.accel.edge_dist.data();

    auto store = [&](int i, int t, int j, float dcc) {
        knn[(size_t)i * m + t] = j;
        edt[(size_t)i * m + t] = dcc;
        const float cni = c_norm[i], cnj = c_norm[j];
        const float denom = std::sqrt(cni * dcc);
        const float cv = (denom > 1e-12f) ? 0.5f * (cni + dcc - cnj) / denom : 0.0f;
        float sv = 1.0f - cv * cv;
        ecs[(size_t)i * m + t] = cv;
        esi[(size_t)i * m + t] = (sv > 0.0f) ? std::sqrt(sv) : 0.0f;
    };
    if (ws.params.opt) {

        const float* Cen = ws.data.Cen;
        #pragma omp parallel
        {
            std::vector<float> dists; dists.assign(k, 0.0f);   
            std::vector<int>   idx;   idx.assign(k, 0);
            #pragma omp for schedule(auto)
            for (int i = 0; i < k; i++) {
                const float* Ci = Cen + (size_t)i * dim;
                const float cni = c_norm[i];
                for (int j = 0; j < k; j++) {
                    idx[j] = j;
                    dists[j] = (j == i) ? 1e30f                                 
                                        : (cni + c_norm[j] - 2.0f * dot_product_512(Ci, Cen + (size_t)j * dim, dim));
                }
                std::nth_element(idx.begin(), idx.begin() + m, idx.end(),
                                 [&](int a, int b) { return dists[a] < dists[b]; });
                for (int t = 0; t < m; t++) { int j = idx[t]; store(i, t, j, dists[j]); }
            }
        }
    } else {

        const float* Dcc = ws.accel.Dcc.data();
        #pragma omp parallel for num_threads(nt) schedule(auto)
        for (int i = 0; i < k; i++) {
            std::vector<int> idx;
            idx.reserve(k > 0 ? k - 1 : 0);
            const float* row = Dcc + (size_t)i * k;
            for (int j = 0; j < k; j++) if (j != i) idx.push_back(j);
            auto closer = [row](int a, int b) { return row[a] < row[b]; };
            if ((int)idx.size() > m) { std::nth_element(idx.begin(), idx.begin() + m, idx.end(), closer); idx.resize(m); }
            std::sort(idx.begin(), idx.end(), closer);
            for (int t = 0; t < m; t++) { int j = idx[t]; store(i, t, j, row[j]); }
        }
    }
}

static void update_fv(Workspace& ws) {
    const int n = ws.params.n, dim = ws.params.dim, k = ws.params.k, nt = ws.params.n_threads;
    const float* X     = ws.data.X;
    const float* Cen   = ws.data.Cen;
    float* c_norm  = ws.data.c_norm.data();
    float* Dcc     = ws.accel.Dcc.data();
    float* r2      = ws.accel.r2.data();
    float* cos_ocx = ws.accel.cos_ocx.data();
    float* sin_ocx = ws.accel.sin_ocx.data();
    const float* x_norm = ws.data.x_norm.data();
    const int* labels   = ws.data.labels.data();

    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int i = 0; i < k; i++)
        c_norm[i] = sum_sq_512(Cen + (size_t)i * dim, dim);

    if (!(ws.params.m > 0 && ws.params.opt)) {   
    constexpr int UF = 16;
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int i0 = 0; i0 < k; i0 += UF) {
        const int mi = (i0 + UF <= k) ? UF : (k - i0);
        const float* A = Cen + (size_t)i0 * dim;
        for (int j = 0; j < k; j++) {
            float dots[16];
            const float* B = Cen + (size_t)j * dim;
            if (mi == UF) {
                micro_16x1_dots(A + (size_t)0*dim,  A + (size_t)1*dim,  A + (size_t)2*dim,  A + (size_t)3*dim,
                                A + (size_t)4*dim,  A + (size_t)5*dim,  A + (size_t)6*dim,  A + (size_t)7*dim,
                                A + (size_t)8*dim,  A + (size_t)9*dim,  A + (size_t)10*dim, A + (size_t)11*dim,
                                A + (size_t)12*dim, A + (size_t)13*dim, A + (size_t)14*dim, A + (size_t)15*dim,
                                B, dim, dots);                                     
            } else {
                for (int m = 0; m < mi; m++) dots[m] = dot_product_512(A + (size_t)m * dim, B, dim);
            }
            const float  cnj  = c_norm[j];
            const size_t base = (size_t)i0 * k + j;                              
            for (int m = 0; m < mi; m++)
                Dcc[base + (size_t)m * k] = c_norm[i0 + m] + cnj - 2.0f * dots[m];
        }
    }
    }

    build_knn(ws);   

    #pragma omp parallel for num_threads(nt) schedule(auto)
    for (int i = 0; i < n; i++) {
        int g = labels[i];
        r2[i] = sum_sq_diff_512(X + (size_t)i * dim, Cen + (size_t)g * dim, dim);
        float cn_g = c_norm[g], r2_i = r2[i], xn_i = x_norm[i];
        float denom = 2.0f * std::sqrt(cn_g * r2_i);
        cos_ocx[i] = (denom > 1e-12f) ? (cn_g + r2_i - xn_i) / denom : 0.0f;
        float sv = 1.0f - cos_ocx[i] * cos_ocx[i];
        sin_ocx[i] = (sv > 0) ? std::sqrt(sv) : 0.0f;
    }
}

constexpr int BM_X_L3 = 512;   
constexpr int BM_X_L2 = 128;   

static inline long bm_l1(uint8_t* row, int cw, int c_off,
                         const float* dccr, const float* cnorm, float cng,
                         float ci, float si, float r2i) {
    long pairs = 0;
    const __m512 cng_b  = _mm512_set1_ps(cng);
    const __m512 cng4_b = _mm512_set1_ps(4.0f * cng);       
    const __m512 cos_b  = _mm512_set1_ps(ci);
    const __m512 sin_b  = _mm512_set1_ps(si);
    const __m512 r2_b   = _mm512_set1_ps(r2i);
    int jj = 0;
    for (; jj + 15 < cw; jj += 16) {
        const int j = c_off + jj;
        const __m512 dcc = _mm512_loadu_ps(dccr + j);        
        const __m512 cnj = _mm512_loadu_ps(cnorm + jj);      
        const __m512 A   = _mm512_add_ps(_mm512_sub_ps(dcc, cnj), cng_b);            
        const __m512 E   = _mm512_max_ps(_mm512_sub_ps(_mm512_mul_ps(cng4_b, dcc),
                                                       _mm512_mul_ps(A, A)), _mm512_setzero_ps());  
        const __m512 num = _mm512_fmadd_ps(A, cos_b, _mm512_mul_ps(sin_b, _mm512_sqrt_ps(E)));
        const __m512 lhs = _mm512_mul_ps(_mm512_mul_ps(num, num), r2_b);             
        const __m512 rhs = _mm512_mul_ps(cng_b, _mm512_mul_ps(dcc, dcc));            
        __mmask16 gt = _mm512_cmp_ps_mask(lhs, rhs, _CMP_GT_OQ);
        uint16_t bits = (uint16_t)gt;
        if (bits) std::memcpy(row + j / 8, &bits, 2);        
        pairs += _mm_popcnt_u32((unsigned)gt);
    }
    for (; jj < cw; jj++) {                                   
        const int j = c_off + jj;
        const float dcc = dccr[j], cnj = cnorm[jj];
        float A = dcc - cnj + cng;
        float E = 4.0f * cng * dcc - A * A;  if (E < 0.0f) E = 0.0f;
        float num = A * ci + si * std::sqrt(E);
        if (num * num * r2i > cng * dcc * dcc) { row[j / 8] |= (uint8_t)(1u << (j % 8)); pairs++; }
    }
    return pairs;
}

static inline long bm_l2(int x_off, int x_rows, int c_off, int cw, int k, int mstride,
                         uint8_t* mask, const float* Dcc, const float* c_norm,
                         const float* cos_ocx, const float* sin_ocx, const float* r2,
                         const int* labels) {
    long pairs = 0;
    const float* cnorm_panel = c_norm + c_off;               
    for (int i = x_off; i < x_off + x_rows; i++) {
        const int g = labels[i];
        pairs += bm_l1(mask + (size_t)i * mstride, cw, c_off,
                       Dcc + (size_t)g * k, cnorm_panel, c_norm[g],
                       cos_ocx[i], sin_ocx[i], r2[i]);
    }
    return pairs;
}

static inline long bm_l3(Workspace& ws, int x_off, int x_rows, int c_off, int cw) {
    const int k = ws.params.k, mstride = ws.params.mstride;
    uint8_t*      mask    = ws.accel.mask_buf.data();
    const float*  Dcc     = ws.accel.Dcc.data();
    const float*  c_norm  = ws.data.c_norm.data();
    const float*  cos_ocx = ws.accel.cos_ocx.data();
    const float*  sin_ocx = ws.accel.sin_ocx.data();
    const float*  r2      = ws.accel.r2.data();
    const int*    labels  = ws.data.labels.data();
    long pairs = 0;
    for (int p_off = 0; p_off < x_rows; p_off += BM_X_L2) {
        const int p_rows = std::min(BM_X_L2, x_rows - p_off);
        pairs += bm_l2(x_off + p_off, p_rows, c_off, cw, k, mstride,
                       mask, Dcc, c_norm, cos_ocx, sin_ocx, r2, labels);
    }
    return pairs;
}

static long build_mask(Workspace& ws) {
    const int n = ws.params.n, k = ws.params.k, mstride = ws.params.mstride, nt = ws.params.n_threads;
    uint8_t*   mask   = ws.accel.mask_buf.data();
    const int* labels = ws.data.labels.data();

    int CW = (int)(16'000'000 / (4.0 * k));
    CW = (CW / 16) * 16;  if (CW < 16) CW = 16;  if (CW > 512) CW = 512;

    long pairs = 0;

    #pragma omp parallel num_threads(nt)
    {
        long local_pairs = 0;                              

        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
            std::memset(mask + (size_t)i * mstride, 0, mstride);

        for (int c_off = 0; c_off < k; c_off += CW) {
            const int cw = std::min(CW, k - c_off);
            #pragma omp for schedule(static)
            for (int x_off = 0; x_off < n; x_off += BM_X_L3) {
                const int x_rows = std::min(BM_X_L3, n - x_off);
                local_pairs += bm_l3(ws, x_off, x_rows, c_off, cw);
            }
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            const int g = labels[i];
            uint8_t* row = mask + (size_t)i * mstride;
            if (!((row[g / 8] >> (g % 8)) & 1)) { row[g / 8] |= (uint8_t)(1u << (g % 8)); local_pairs++; }
        }

        #pragma omp atomic
        pairs += local_pairs;                              
    }
    return pairs;
}

static long build_mask_graph(Workspace& ws) {
    const int n = ws.params.n, k = ws.params.k, m = ws.params.m, mstride = ws.params.mstride, nt = ws.params.n_threads;
    uint8_t*     mask   = ws.accel.mask_buf.data();
    const float* Dcc    = ws.accel.Dcc.data();
    const float* cos_ocx= ws.accel.cos_ocx.data();
    const float* sin_ocx= ws.accel.sin_ocx.data();
    const float* r2     = ws.accel.r2.data();
    const int*   labels = ws.data.labels.data();
    const int*   knn    = ws.accel.knn_idx.data();
    const float* ecs    = ws.accel.edge_cos.data();      
    const float* esi    = ws.accel.edge_sin.data();
    const float* edt    = ws.accel.edge_dist.data();     
    const int    opt    = ws.params.opt;
    long pairs = 0;
    #pragma omp parallel for num_threads(nt) schedule(auto) reduction(+:pairs)
    for (int i = 0; i < n; i++) {
        const int g = labels[i];
        uint8_t* row = mask + (size_t)i * mstride;
        std::memset(row, 0, mstride);                    
        const float ci = cos_ocx[i], si = sin_ocx[i], r2x4 = 4.0f * r2[i];
        const int*   nbrs  = knn + (size_t)g * m;
        const float* erowc = ecs  + (size_t)g * m;
        const float* erows = esi  + (size_t)g * m;
        for (int t = 0; t < m; t++) {
            const int j = nbrs[t];                       
            const float cv = erowc[t] * ci + erows[t] * si;     
            const float thresh = opt ? edt[(size_t)g * m + t] : Dcc[(size_t)g * k + j];  
            if (cv * cv * r2x4 > thresh) { row[j / 8] |= (uint8_t)(1u << (j % 8)); pairs++; }
        }
        if (!((row[g / 8] >> (g % 8)) & 1)) { row[g / 8] |= (uint8_t)(1u << (g % 8)); pairs++; }  
    }
    return pairs;
}

static long build_cand_list(Workspace& ws) {
    const int n = ws.params.n, k = ws.params.k, m = ws.params.m, nt = ws.params.n_threads;
    const int cap = m + 1;
    const float* Dcc     = ws.accel.Dcc.data();
    const float* cos_ocx = ws.accel.cos_ocx.data();
    const float* sin_ocx = ws.accel.sin_ocx.data();
    const float* r2      = ws.accel.r2.data();
    const float* ecs     = ws.accel.edge_cos.data();
    const float* esi     = ws.accel.edge_sin.data();
    const float* edt     = ws.accel.edge_dist.data();
    const int*   labels  = ws.data.labels.data();
    const int*   knn     = ws.accel.knn_idx.data();
    int*   cand_idx = ws.accel.cand_idx.data();
    int*   cand_cnt = ws.accel.cand_cnt.data();
    const int opt = ws.params.opt;
    long pairs = 0;
    #pragma omp parallel for num_threads(nt) schedule(auto) reduction(+:pairs)
    for (int i = 0; i < n; i++) {
        const int g = labels[i];
        int* cand = cand_idx + (size_t)i * cap;
        const float ci = cos_ocx[i], si = sin_ocx[i], r2x4 = 4.0f * r2[i];
        const int*   nbrs  = knn + (size_t)g * m;
        const float* erowc = ecs  + (size_t)g * m;
        const float* erows = esi  + (size_t)g * m;
        const float* edg   = edt  + (size_t)g * m;
        int cnt = 0;
        for (int t = 0; t < m; t++) {
            const int j = nbrs[t];
            const float cv = erowc[t] * ci + erows[t] * si;
            const float thresh = opt ? edg[t] : Dcc[(size_t)g * k + j];
            if (cv * cv * r2x4 > thresh) cand[cnt++] = j;
        }
        cand[cnt++] = g;            
        cand_cnt[i] = cnt;
        pairs += cnt;
    }
    return pairs;
}

} 
#endif  

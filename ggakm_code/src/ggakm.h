#ifndef AKM_MYKM_GGAKM_H
#define AKM_MYKM_GGAKM_H

#include <omp.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>

#include "kernels.h"    
#include "workspace.h"
#include "accel.h"      
#include "assign.h"     
#include "update.h"     

namespace mykm {

static inline void refresh_graph(Workspace& ws, Timings& tm) {
    const int k = ws.params.k, dim = ws.params.dim, nt = ws.params.n_threads;
    double t0 = omp_get_wtime();
    row_norm_sq(ws.data.Cen, ws.data.c_norm.data(), k, dim, nt);   
    build_knn(ws);                                                  
    tm.update_fv += omp_get_wtime() - t0;
}

static int run_ggakm(Workspace& ws, Timings& tm) {
    const int n   = ws.params.n;
    const int k   = ws.params.k;
    const int dim = ws.params.dim;
    const int m   = ws.params.m;
    const int nt  = ws.params.n_threads;
    const int b   = ws.params.batch_size > 0 ? ws.params.batch_size : 1024;
    ws.params.opt = 1;   

    float*       Cen     = ws.data.Cen;
    const float* X       = ws.data.X;
    const float* x_norm  = ws.data.x_norm.data();
    int*         labels  = ws.data.labels.data();
    float*       cnt     = ws.data.cnt.data();
    const int*   knn     = ws.accel.knn_idx.data();
    const float* edge_cos = ws.accel.edge_cos.data();
    const float* edge_sin = ws.accel.edge_sin.data();
    const float* edge_dist= ws.accel.edge_dist.data();

    const double t_loop = omp_get_wtime();

    {
        reset_accumulators(ws);
        double t0 = omp_get_wtime();
        e_step(ws, false);                  
        tm.e_step += omp_get_wtime() - t0;
    }

    refresh_graph(ws, tm);

    std::memset(cnt, 0, (size_t)k * sizeof(float));
    for (int i = 0; i < n; i++) cnt[labels[i]] += 1.0f;

    const int    batches_per_epoch = (n + b - 1) / b;   

    const int    refresh_every     = ws.params.refresh_period > 0 ? ws.params.refresh_period : batches_per_epoch;
    const int    n_steps           = (ws.params.max_iter * n) / b;        

    const int    patience          = ws.params.max_no_improvement > 0 ? ws.params.max_no_improvement : 10;

    const double alpha             = std::min(1.0, (double)b * 2.0 / ((double)n + 1.0));

    uint64_t rng = 0x2545F4914F6CDD1Dull;
    auto next_idx = [&]() -> int {
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        uint64_t x = rng * 0x2545F4914F6CDD1Dull;
        return (int)((x >> 32) % (uint32_t)n);
    };

    std::vector<int>   batch_idx(b);
    std::vector<int>   batch_cstar(b);    
    std::vector<float> batch_dmin(b);     

    double ewa_inertia     = 0.0;
    double ewa_inertia_min = std::numeric_limits<double>::infinity();
    bool   ewa_init        = false;
    int    no_improvement  = 0;
    int    refresh_counter = 0;
    int    n_iter          = 0;           
    long   pairs_total     = 0;           

    for (int it = 0; it < n_steps; ++it) {

        for (int s = 0; s < b; s++) batch_idx[s] = next_idx();

        double t0 = omp_get_wtime();
        long   pairs_batch = 0;
        #pragma omp parallel for num_threads(nt) schedule(static) reduction(+:pairs_batch)
        for (int s = 0; s < b; s++) {
            const int i = batch_idx[s];
            const int g = labels[i];
            const float* xi = X + (size_t)i * dim;
            const float* cg = Cen + (size_t)g * dim;

            const float r2_i = sum_sq_diff_512(xi, cg, dim);          
            const float cn_g = sum_sq_512(cg, dim);                   
            float denom = 2.0f * std::sqrt(cn_g * r2_i);
            float ci = (denom > 1e-12f) ? (cn_g + r2_i - x_norm[i]) / denom : 0.0f;
            float sv = 1.0f - ci * ci; float si = (sv > 0.0f) ? std::sqrt(sv) : 0.0f;
            const float r2x4 = 4.0f * r2_i;

            float best = r2_i; int bj = g;
            long  pcnt = 1;                                          
            const int*   nbrs = knn + (size_t)g * m;
            const float* erc  = edge_cos  + (size_t)g * m;
            const float* ers  = edge_sin  + (size_t)g * m;
            const float* erd  = edge_dist + (size_t)g * m;
            for (int t = 0; t < m; t++) {
                const int j = nbrs[t];
                float cv = erc[t] * ci + ers[t] * si;
                if (cv * cv * r2x4 > erd[t]) {                        
                    float d = sum_sq_diff_512(xi, Cen + (size_t)j * dim, dim);
                    pcnt++;
                    if (d < best) { best = d; bj = j; }
                }
            }
            batch_cstar[s] = bj;
            batch_dmin[s]  = best;                                    
            labels[i]      = bj;                                      
            pairs_batch   += pcnt;
        }
        tm.e_step += omp_get_wtime() - t0;
        pairs_total += pairs_batch;

        t0 = omp_get_wtime();
        double batch_inertia = 0.0;

        for (int s = 0; s < b; s++) {
            const int i  = batch_idx[s];
            const int cs = batch_cstar[s];
            batch_inertia += (double)batch_dmin[s];   
            cnt[cs] += 1.0f;                            
            float eta = 1.0f / cnt[cs];                 
            float* c = Cen + (size_t)cs * dim;
            const float* xi = X + (size_t)i * dim;
            for (int d = 0; d < dim; d++) { float df = xi[d] - c[d]; c[d] += eta * df; }  
        }
        tm.m_step += omp_get_wtime() - t0;

        if (++refresh_counter >= refresh_every) { refresh_graph(ws, tm); refresh_counter = 0; }

        n_iter = it + 1;          

        if (it == 0) continue;

        double mean_inertia = batch_inertia / (double)b;                       
        if (!ewa_init) { ewa_inertia = mean_inertia; ewa_init = true; }        
        else           ewa_inertia = ewa_inertia * (1.0 - alpha) + mean_inertia * alpha;  
        if (ewa_inertia < ewa_inertia_min) { ewa_inertia_min = ewa_inertia; no_improvement = 0; } 
        else                               no_improvement++;                  
        if (no_improvement >= patience) break;                                
    }
    tm.total_pairs += pairs_total;

    {
        row_norm_sq(ws.data.Cen, ws.data.c_norm.data(), k, dim, nt);
        reset_accumulators(ws);
        double t0 = omp_get_wtime();
        e_step(ws, false);                  
        tm.e_step += omp_get_wtime() - t0;
    }

    tm.total = omp_get_wtime() - t_loop;
    return n_iter;
}

} 
#endif  

#ifndef AKM_MYKM_UPDATE_H
#define AKM_MYKM_UPDATE_H

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "workspace.h"

namespace mykm {

static inline void reset_accumulators(Workspace& ws) {
    std::memset(ws.data.wc_threads.data(),  0,
                (size_t)ws.params.n_threads * ws.params.k * sizeof(float));
    std::memset(ws.data.Cen_threads.data(), 0,
                (size_t)ws.params.n_threads * ws.params.k * ws.params.dim * sizeof(float));
}

static float m_step(Workspace& ws) {
    const int k = ws.params.k, dim = ws.params.dim, nt = ws.params.n_threads, n = ws.params.n;
    const float* X       = ws.data.X;
    const float* Cen_old = ws.data.Cen;
    float*       Cen_new = ws.data.Cen_new.data();
    const float* Cen_t   = ws.data.Cen_threads.data();   
    const float* wc_t    = ws.data.wc_threads.data();    
    float* wc            = ws.data.wc.data();

    #pragma omp parallel for num_threads(nt) schedule(auto)
    for (int j = 0; j < k; j++) {
        float w = 0.0f;
        for (int t = 0; t < nt; t++) w += wc_t[(size_t)t * k + j];
        wc[j] = w;
        if (w > 0.0f) {
            const float inv = 1.0f / w;
            float* cn = Cen_new + (size_t)j * dim;
            for (int d = 0; d < dim; d++) {
                float acc = 0.0f;
                for (int t = 0; t < nt; t++) acc += Cen_t[((size_t)t * k + j) * dim + d];
                cn[d] = acc * inv;
            }
        }
    }

    ws.data.empties.clear();
    for (int j = 0; j < k; j++) if (wc[j] == 0.0f) ws.data.empties.push_back(j);
    const int E0 = (int)ws.data.empties.size();
    if (E0 > 0) {
        const int E = std::min(E0, n);          
        DistIdx* buf = ws.data.reinit_buf.data();
        for (int i = 0; i < n; i++) buf[i] = {ws.data.dist_sq[i], i};
        std::nth_element(buf, buf + E, buf + n,
                         [](const DistIdx& a, const DistIdx& b) { return a.d > b.d; });
        for (int e = 0; e < E; e++) {           
            const int idx = buf[e].i;
            std::memcpy(Cen_new + (size_t)ws.data.empties[e] * dim,
                        X + (size_t)idx * dim, (size_t)dim * sizeof(float));
        }
        for (int e = E; e < E0; e++) {          
            std::memcpy(Cen_new + (size_t)ws.data.empties[e] * dim,
                        Cen_old + (size_t)ws.data.empties[e] * dim, (size_t)dim * sizeof(float));
        }
    }

    float shift = 0.0f;
    #pragma omp parallel for num_threads(nt) schedule(auto) reduction(max:shift)
    for (int j = 0; j < k; j++) {
        const float* old = Cen_old + (size_t)j * dim;
        const float* nw  = Cen_new + (size_t)j * dim;
        float s = 0.0f;
        for (int d = 0; d < dim; d++) { float df = nw[d] - old[d]; s += df * df; }
        shift = std::max(shift, std::sqrt(s));
    }
    return shift;
}

} 
#endif  

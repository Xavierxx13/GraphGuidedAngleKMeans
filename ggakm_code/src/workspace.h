#ifndef AKM_MYKM_WORKSPACE_H
#define AKM_MYKM_WORKSPACE_H

#include <vector>
#include <cstdint>
#include <utility>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kernels.h"   

namespace mykm {

struct DistIdx { float d; int i; };

struct Params {
    int   n;          
    int   dim;        
    int   k;          
    int   n_threads;  
    int   max_iter;   
    float tol;        
    int   mstride;    
    int   m = 0;      
    int   opt = 0;    

    int   batch_size = 0;          
    int   refresh_period = 0;      
    int   max_no_improvement = 10; 
};

struct Data {

    const float* X   = nullptr;   
    float*       Cen = nullptr;   

    std::vector<float> c_norm;    
    std::vector<float> x_norm;    
    std::vector<float> dist_sq;   

    std::vector<int>   labels;    

    std::vector<float> wc_threads;   
    std::vector<float> Cen_threads;  

    std::vector<float>    wc;          
    std::vector<float>    Cen_new;     
    std::vector<float>    cnt;         
    std::vector<int>      empties;     
    std::vector<DistIdx>  reinit_buf;  
};

struct Accel {

    std::vector<uint8_t> mask_buf;   
    const uint8_t* mask = nullptr;   

    std::vector<float> Dcc;          

    std::vector<float> cos_ocx;      
    std::vector<float> sin_ocx;      
    std::vector<float> r2;           

    std::vector<int>   knn_idx;      
    std::vector<float> edge_cos;     
    std::vector<float> edge_sin;     
    std::vector<float> edge_dist;    
    std::vector<int>   cand_idx;     
    std::vector<int>   cand_cnt;     
};

struct Workspace {
    Params params;
    Data   data;
    Accel  accel;

    Workspace(pybind11::array_t<float> X, pybind11::array_t<float> Cen,
              int max_iter, int n_threads, float tol, int m_ = 0, int opt_ = 0) {

        const int n   = (int)X.shape(0);
        const int dim = (int)X.shape(1);
        const int k   = (int)Cen.shape(0);
        const int mstride = ((k + 63) / 64) * 8;
        params   = {n, dim, k, n_threads, max_iter, tol, mstride};
        params.m = (m_ > 0 && m_ < k) ? m_ : 0;   
        params.opt = (params.m > 0) ? opt_ : 0;   

        data.X   = (const float*)X.unchecked().data(0);
        data.Cen = (float*)Cen.mutable_unchecked().mutable_data(0);

        data.c_norm.assign(k, 0.0f);
        data.x_norm.assign(n, 0.0f);
        data.dist_sq.assign(n, 0.0f);
        data.labels.assign(n, 0);
        data.wc_threads.assign((size_t)n_threads * k, 0.0f);
        data.Cen_threads.assign((size_t)n_threads * k * dim, 0.0f);
        data.wc.assign(k, 0.0f);
        data.Cen_new.assign((size_t)k * dim, 0.0f);
        data.cnt.assign(k, 0.0f);
        data.empties.reserve(k);
        data.reinit_buf.assign(n, DistIdx{0.0f, 0});

        accel.Dcc.assign((size_t)k * k, 0.0f);
        accel.cos_ocx.assign(n, 0.0f);
        accel.sin_ocx.assign(n, 0.0f);
        accel.r2.assign(n, 0.0f);
        accel.knn_idx.assign((size_t)k * (params.m > 0 ? params.m : 1), 0);   
        accel.edge_cos.assign((size_t)k * (params.m > 0 ? params.m : 1), 0.0f);
        accel.edge_sin.assign((size_t)k * (params.m > 0 ? params.m : 1), 0.0f);
        accel.edge_dist.assign((size_t)k * (params.m > 0 ? params.m : 1), 0.0f);
        const int cap = (params.m > 0) ? params.m + 1 : 2;
        accel.cand_idx.assign((size_t)n * cap, 0);
        accel.cand_cnt.assign(n, 0);

        accel.mask_buf.assign((size_t)n * mstride, 0);
        for (int i = 0; i < n; i++) {
            uint8_t* row = accel.mask_buf.data() + (size_t)i * mstride;
            int full = k / 8;
            for (int b = 0; b < full; b++) row[b] = 0xFFu;
            if (k % 8) row[full] = (uint8_t)((1u << (k % 8)) - 1u);
        }
        accel.mask = accel.mask_buf.data();

        row_norm_sq(data.X,   data.x_norm.data(), n, dim, n_threads);
        row_norm_sq(data.Cen, data.c_norm.data(), k, dim, n_threads);
    }
};

struct Timings {
    double e_step   = 0;   
    double m_step   = 0;   
    double build    = 0;   
    double update_fv = 0;  
    double total    = 0;   
    long   total_pairs = 0;  
};

} 
#endif  

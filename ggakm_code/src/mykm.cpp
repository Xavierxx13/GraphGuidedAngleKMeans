
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <cstdint>

#include "workspace.h"
#include "assign.h"    
#include "update.h"    
#include "accel.h"     
#include "ggakm.h"     

namespace py = pybind11;
using namespace mykm;

template <typename T> inline T* mut_ptr(py::array_t<T>& a) { return a.mutable_unchecked().mutable_data(0); }

static double dense_thresh() {
    static const double t = []{ const char* e = std::getenv("MYKM_DENSE_THRESH");
                                return e ? std::atof(e) : 0.30; }();
    return t;
}

static int run(Workspace& ws, Timings& tm) {
    const int n = ws.params.n, k = ws.params.k;
    float* cur     = ws.data.Cen;
    float* scratch = ws.data.Cen_new.data();
    const size_t cen_bytes = (size_t)k * ws.params.dim * sizeof(float);

    const double t_loop = omp_get_wtime();
    int n_iter = 0;
    for (int it = 0; it < ws.params.max_iter; ++it) {
        reset_accumulators(ws);

        double t0;
        long   pairs;
        bool   use_dense;
        if (it > 0) {
            t0 = omp_get_wtime();
            if (ws.params.m > 0) {                    
                pairs = build_cand_list(ws);
                tm.build += omp_get_wtime() - t0;
                use_dense = false;                    
            } else {                                  
                pairs = build_mask(ws);
                tm.build += omp_get_wtime() - t0;
                use_dense = (double)pairs > dense_thresh() * (double)n * (double)k;
                if (use_dense) pairs = (long)n * k;      
            }
        } else {
            use_dense = true;                         
            pairs = (long)n * k;
        }
        tm.total_pairs += pairs;

        t0 = omp_get_wtime();
        if (ws.params.m > 0 && it > 0) e_step_list(ws);   
        else                          e_step(ws, use_dense);  
        tm.e_step += omp_get_wtime() - t0;

        t0 = omp_get_wtime();
        float shift = m_step(ws);
        tm.m_step += omp_get_wtime() - t0;

        std::memcpy(cur, scratch, cen_bytes);     
        n_iter = it + 1;
        if (ws.params.tol >= 0.0f && shift <= ws.params.tol) break; 

        t0 = omp_get_wtime();
        update_fv(ws);
        tm.update_fv += omp_get_wtime() - t0;
    }
    tm.total = omp_get_wtime() - t_loop;
    return n_iter;
}

static py::tuple mykm_impl(py::array_t<float> X_arr, py::array_t<float> Cen_arr,
                           int max_iter, int n_threads, float tol, int m, int opt) {

    Workspace ws(X_arr, Cen_arr, max_iter, n_threads, tol, m, opt);

    Timings tm;
    int n_iter = run(ws, tm);

    py::array_t<int> labels_out(ws.params.n);
    std::memcpy(mut_ptr(labels_out), ws.data.labels.data(),
                (size_t)ws.params.n * sizeof(int));

    py::dict timing;
    timing["e_step_ms"]    = tm.e_step    * 1e3;
    timing["m_step_ms"]    = tm.m_step    * 1e3;
    timing["build_ms"]     = tm.build     * 1e3;   
    timing["update_fv_ms"] = tm.update_fv * 1e3;   
    timing["total_ms"]     = tm.total     * 1e3;
    timing["n_iter"]       = n_iter;
    timing["total_pairs"]  = (double)tm.total_pairs;
    return py::make_tuple(labels_out, n_iter, timing);
}

static py::tuple ggakm_impl(py::array_t<float> X_arr, py::array_t<float> Cen_arr,
                            int max_iter, int n_threads, float tol, int m, int opt,
                            int batch_size, int refresh_period, int max_no_improvement) {

    const int kdim = (int)Cen_arr.shape(0);
    int eff_m = (m > 0 && m < kdim) ? m : (kdim > 5 ? 5 : (kdim > 1 ? kdim - 1 : 1));

    Workspace ws(X_arr, Cen_arr, max_iter, n_threads, tol, eff_m, opt);
    ws.params.batch_size         = batch_size > 0 ? batch_size : 1024;
    ws.params.refresh_period     = refresh_period;
    ws.params.max_no_improvement = max_no_improvement > 0 ? max_no_improvement : 10;

    Timings tm;
    int n_iter = run_ggakm(ws, tm);

    py::array_t<int> labels_out(ws.params.n);
    std::memcpy(mut_ptr(labels_out), ws.data.labels.data(),
                (size_t)ws.params.n * sizeof(int));

    py::dict timing;
    timing["e_step_ms"]    = tm.e_step    * 1e3;
    timing["m_step_ms"]    = tm.m_step    * 1e3;
    timing["build_ms"]     = tm.build     * 1e3;
    timing["update_fv_ms"] = tm.update_fv * 1e3;   
    timing["total_ms"]     = tm.total     * 1e3;
    timing["n_iter"]       = n_iter;               
    timing["total_pairs"]  = (double)tm.total_pairs;
    return py::make_tuple(labels_out, n_iter, timing);
}

PYBIND11_MODULE(_mykm, m) {
    m.doc() = "GGAKM: graph-guided angle k-means.";
    m.def("mykm", &mykm_impl,
          py::arg("X"), py::arg("Cen"),
          py::arg("max_iter") = 300, py::arg("n_threads") = 64, py::arg("tol") = 1e-4f, py::arg("m") = 0, py::arg("opt") = 0,
          "Angle k-means. X: (n, dim) float32, Cen: (k, dim) float32, updated in place."
          " Returns (labels, n_iter, timings).");
    m.def("ggakm", &ggakm_impl,
          py::arg("X"), py::arg("Cen"),
          py::arg("max_iter") = 100, py::arg("n_threads") = 64, py::arg("tol") = 1e-4f,
          py::arg("m") = 5, py::arg("opt") = 1,
          py::arg("batch_size") = 1024, py::arg("refresh_period") = 0, py::arg("max_no_improvement") = 10,
          "GGAKM. X: (n, dim) float32, Cen: (k, dim) float32, updated in place."
          " m: number of graph neighbors. batch_size: samples per batch."
          " refresh_period: graph rebuild period in batches (0 = once per epoch)."
          " max_no_improvement: early-stopping patience. Returns (labels, n_iter, timings).");
}

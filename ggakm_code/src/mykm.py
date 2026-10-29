import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mykm


def _align64(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ctypes.data % 64 == 0:
        return x
    flat = np.empty(x.size + 16, dtype=np.float32)
    off = (-flat.ctypes.data) % 64 // 4
    out = flat[off:off + x.size].reshape(x.shape)
    out[...] = x
    return out


class GGAKM:

    def __init__(self, max_iter=100, tol=1e-4, n_threads=64, m=5, opt=1,
                 batch_size=1024, refresh_period=0, max_no_improvement=10):
        self.max_iter = max_iter
        self.tol = tol
        self.n_threads = n_threads
        self.m = m
        self.opt = opt
        self.batch_size = batch_size
        self.refresh_period = refresh_period
        self.max_no_improvement = max_no_improvement

    def fit(self, X, Cen):
        X = _align64(np.asarray(X, dtype=np.float32))
        Cen = _align64(np.asarray(Cen, dtype=np.float32))
        if Cen.shape[1] != X.shape[1]:
            raise ValueError("X and Cen must have the same feature dimension")
        labels, n_iter, timings = _mykm.ggakm(
            X, Cen, self.max_iter, self.n_threads, self.tol, self.m, self.opt,
            self.batch_size, self.refresh_period, self.max_no_improvement)
        self.labels_ = labels
        self.cluster_centers_ = Cen
        self.n_iter_ = n_iter
        self.timings_ = timings
        return self

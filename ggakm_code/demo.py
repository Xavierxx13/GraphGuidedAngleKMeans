import os

n_threads = 64
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)

import sys
import time

import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import mykm

name = "LFW"
k = 1024
repeats = 10
m = 8
r = 2
batch_size = 1024
max_iter = 100
patience = 10

data_path = os.path.join(os.path.dirname(__file__), "demo_data", name + ".mat")
centroids_path = os.path.join(os.path.dirname(__file__), "demo_data",
                              name + "_k" + str(k) + "_centroids.mat")

data = sio.loadmat(data_path)
X = np.ascontiguousarray(data["X"], dtype=np.float32)
y_true = np.asarray(data["y_true"]).ravel()
n, d = X.shape

center_data = sio.loadmat(centroids_path)
C = np.ascontiguousarray(center_data["centroids"], dtype=np.float32)


def bench(fn):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mod = fn()
        ts.append(time.perf_counter() - t0)
    return sum(ts) / len(ts), mod


print("")
print("Data Info")
print("n = " + str(n))
print("d = " + str(d))
print("k = " + str(k))
print("m = " + str(m))
print("r = " + str(r))

print("")
print("Lloyd")
t, mod = bench(lambda: KMeans(n_clusters=k, init=C.copy(), n_init=1, max_iter=300,
                              tol=1e-4, algorithm="lloyd", random_state=0).fit(X))
print("time = " + str(round(t, 3)) + " s")
print("NMI = " + str(round(normalized_mutual_info_score(y_true, mod.labels_), 4)))

time.sleep(2)
print("")
print("GGAKM")
bpe = int(np.ceil(n / batch_size))
t, mod = bench(lambda: mykm.GGAKM(max_iter=max_iter, n_threads=n_threads, m=m, opt=1,
                                  batch_size=batch_size, refresh_period=r * bpe,
                                  max_no_improvement=patience).fit(X, C.copy()))
print("time = " + str(round(t, 3)) + " s")
print("NMI = " + str(round(normalized_mutual_info_score(y_true, mod.labels_), 4)))

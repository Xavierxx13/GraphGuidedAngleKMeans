# GGAKM



## Code Repository

ggakm_code/

├── demo.py                   # demonstration script

├── demo_data/

│  ├── LFW.mat                           # sample dataset

│  └── LFW_k1024_centroids.mat  # sample centroid file

└── src/

  ├── build.py               # compilation script

  ├── mykm.py                # python interface

  ├── mykm.cpp               # C++ source

  └── accel.h, assign.h, ggakm.h, kernels.h, update.h, workspace.h

This repository provides a demonstration script (*demo.py*), a sample dataset (*LFW.mat*), and its centroid file at k=1024 (*LFW_k1024_centroids.mat*). The demo compares Lloyd and GGAKM on the same data and the same initial centroids, and reports the running time and NMI of each method.



### Environment Setup

The C++ part is compiled locally, so a Linux (x86-64) machine with g++ (version 9 or newer) and an AVX-512 capable CPU is required.

Our running environment:

- OS: Ubuntu 24.04 LTS
- CPU: AMD EPYC 9755
- GPU: NVIDIA RTX PRO 6000
- Memory: 314 GB

1. Create a new Anaconda environment named py13 with Python 3.13 using the command:

   ```bash
   (base) $ conda create -n py13 python=3.13
   ```

2. Activate the environment using the command:

   ```bash
   (base) $ conda activate py13
   ```

   After successful activation, your command line prompt should show something like:

   ```bash
   (py13) $
   ```



### Preparation & Installation

1. Install the necessary libraries using the following command:

   ```bash
   (py13) $ pip install numpy scipy scikit-learn pybind11
   ```

2. Access the *src* directory (replace /path/to/ggakm_code with the actual path on your machine):

   ```bash
   (py13) $ cd /path/to/ggakm_code/src
   ```

3. Compile the C++ extension. This produces a file named *_mykm...so* in the same directory:

   ```bash
   (py13) $ python build.py
   ```



### Running the Demo

Go back to the *ggakm_code* directory and execute:

```bash
(py13) $ cd /path/to/ggakm_code
(py13) $ python demo.py
```



### Expected Output

For the dataset *LFW.mat* and its centroid file *LFW\_k1024\_centroids.mat*:

- Each method is run 30 times and the mean running time is reported.
- All methods use 64 threads.
- Running times vary with the machine and its load.
- The NMI values are deterministic.

The expected output is shown below.

```
Data Info
n = 11200
d = 256
k = 1024
m = 8
r = 2

Lloyd
time = 0.104 s
NMI = 0.8507

GGAKM
time = 0.026 s
NMI = 0.8478
```



# Data Introduction

### Data Files

*ggakm_code* provides a sample face embedding file, *LFW.mat*, where the number of data points (*n*) is 11200 and the data dimensionality (*d*) is 256. The following content from the data file is readable.

| Parameter | Description                       |
| --------- | --------------------------------- |
| X         | Data points (shape: [n, d])       |
| y_true    | Ground-truth labels (shape: [n])  |

### Centroid Files

Since k-means involves randomness, a fixed set of initial centroids is used in the demo for reproducibility. The file *LFW\_k1024\_centroids.mat* contains the centroids for *LFW.mat* with k=1024. The centroids are determined by *kmeans++* (seed=0) followed by 3 Lloyd iterations. The following content from the centroid file is readable.

| Parameter | Description                       |
| --------- | --------------------------------- |
| centroids | Cluster centroids (shape: [k, d]) |
| seed      | Random seed (0)                   |
| n         | Number of data points             |
| d         | Dimensionality of the data points |
| k         | Number of clusters                |

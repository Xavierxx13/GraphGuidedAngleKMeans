import os
import subprocess
import sys


def build():
    cur = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(cur, "mykm.cpp")
    out = os.path.join(cur, "_mykm")
    import pybind11
    import numpy as np
    import sysconfig
    inc = sysconfig.get_path("include")
    npinc = np.get_include()
    pyinc = pybind11.get_include()
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    cmd = ["g++", "-O3", "-march=native", "-fopenmp", "-std=c++17", "-fPIC",
           "-flto", "-funroll-loops", "-fno-math-errno",
           "-I" + pyinc, "-I" + inc, "-I" + npinc,
           "-shared", src, "-o", out + suffix]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("Build success: " + out + suffix)


if __name__ == "__main__":
    build()

#!/usr/bin/env python3

import numpy as np
import torch
import sys
import math
from einops import rearrange
import os
from pathlib import Path
import glob

np.random.seed(0xCA7CAFE)

script_dir = os.path.dirname(os.path.realpath(__file__))


def write_example(batch_size, seqlen, nheads, headdim):
    prefix = os.path.join(script_dir, f"test_{batch_size}x{seqlen}x{nheads}x{headdim}")

    q_fname = f"{prefix}_q.bin"
    k_fname = f"{prefix}_k.bin"
    v_fname = f"{prefix}_v.bin"
    if Path(q_fname).is_file() and Path(k_fname).is_file() and Path(v_fname).is_file():
        print(f"Skipping {batch_size}x{seqlen}x{nheads}x{headdim}; already exists")
    else:
        dtype = torch.bfloat16
        with torch.no_grad():
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, dtype=dtype)

        def to_bytes(t):
            with torch.no_grad():
                return t.contiguous().flatten().view(torch.uint8).numpy().tobytes()

        q, k, v = map(to_bytes, qkv.unbind(dim=2))

        with open(q_fname, "wb") as f:
            f.write(q)
        print(f"Wrote {q_fname!r}")

        with open(k_fname, "wb") as f:
            f.write(k)
        print(f"Wrote {k_fname!r}")

        with open(v_fname, "wb") as f:
            f.write(v)
        print(f"Wrote {v_fname!r}")
    return {q_fname, k_fname, v_fname}


fnames = set()
sizes = np.genfromtxt("test_sizes.csv", delimiter=",", ndmin=2, dtype=np.int32)
for row_index in range(sizes.shape[0]):
    fnames |= write_example(*sizes[row_index].tolist())
for fname in glob.glob(os.path.join(script_dir, "test_*.bin")):
    if fname not in fnames:
        print(f"Deleting {fname}; not in test_sizes.csv")
        Path.unlink(Path(fname))

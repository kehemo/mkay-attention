#!/usr/bin/env python3

import numpy as np
import torch
import sys
import math
from einops import rearrange
import os

np.random.seed(0xCA7CAFE)

script_dir = os.path.dirname(os.path.realpath(__file__))


def write_example(batch_size, seqlen, nheads, headdim):

    dtype = torch.bfloat16
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, dtype=dtype)

    def to_bytes(t):
        return bytearray(t.contiguous().flatten().view(torch.uint8))

    q, k, v = map(to_bytes, qkv.unbind(dim=2))

    prefix = os.path.join(script_dir, f"test_{batch_size}x{seqlen}x{nheads}x{headdim}")

    q_fname = f"{prefix}_q.bin"
    k_fname = f"{prefix}_k.bin"
    v_fname = f"{prefix}_v.bin"
    with open(q_fname, "wb") as f:
        f.write(q)
    print(f"Wrote {q_fname!r}")

    with open(k_fname, "wb") as f:
        f.write(k)
    print(f"Wrote {k_fname!r}")

    with open(v_fname, "wb") as f:
        f.write(v)
    print(f"Wrote {v_fname!r}")


write_example(2, 32, 64, 32)

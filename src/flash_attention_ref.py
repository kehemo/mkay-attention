from flash_attn import flash_attn_func
import torch
import os
import math
from einops import rearrange
from ctypes import *
import sys
import numpy as np


file_path = os.path.realpath(__file__)
script_dir = os.path.dirname(file_path)
out_dir = os.path.join(script_dir, "out")


def run_ref(batch_size, seqlen, nheads, headdim):
    dtype = torch.bfloat16

    def from_file(fname, dims=(batch_size, seqlen, nheads, headdim)):
        with open(fname, "rb") as f:
            data = f.read()
            buf = (c_char * len(data)).from_buffer_copy(data)
            return torch.frombuffer(buf, dtype=dtype).reshape(dims).to(device="cuda")

    test_name = f"test_{batch_size}x{seqlen}x{nheads}x{headdim}"
    prefix = os.path.join(script_dir, test_name)

    o_fname = os.path.join(out_dir, f"{test_name}_o.bin")
    q_fname = f"{prefix}_q.bin"
    k_fname = f"{prefix}_k.bin"
    v_fname = f"{prefix}_v.bin"
    q, k, v = map(from_file, [q_fname, k_fname, v_fname])
    out = flash_attn_func(q, k, v)

    def to_bytes(t):
        with torch.no_grad():
            return (
                t.to(device="cpu")
                .contiguous()
                .flatten()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )

    o_bytes = to_bytes(out)
    with open(o_fname, "wb") as f:
        f.write(o_bytes)


sizes = np.genfromtxt(
    os.path.join(script_dir, "test_sizes.csv"), delimiter=",", ndmin=2, dtype=np.int32
)
for row_index in range(sizes.shape[0]):
    run_ref(*sizes[row_index].tolist())

import io
import torch
from dataclasses import dataclass
import triton_naive_attention
from ctypes import *

def from_file(fname, dtype, dims):
        with open(fname, "rb") as f:
            data = f.read()
            buf = (c_char * len(data)).from_buffer_copy(data)
            return torch.frombuffer(buf, dtype=dtype).reshape(dims)

@dataclass
class TestConfig:
    batch_size: int
    seqlen: int
    nheads: int
    headdim: int

def main():
    configs = [TestConfig(2, 32, 64, 32)]
    dtype = torch.bfloat16

    for config in configs:
        test_pref = f"test_{config.batch_size}x{config.seqlen}x{config.nheads}x{config.headdim}"
        dims = (config.batch_size, config.seqlen, config.nheads, config.headdim)
        q = from_file(f"{test_pref}_q.bin", dtype, dims)
        k = from_file(f"{test_pref}_k.bin", dtype, dims)
        v = from_file(f"{test_pref}_v.bin", dtype, dims)
        qkv = torch.stack((q, k, v), dim=2)
        qkv = qkv.to("cuda")
        
        result = triton_naive_attention.attention_triton_launch(qkv)
        
        out = result.cpu()
        out_float32 = out.to(torch.float32)
        out_float32.numpy().tofile(f"out/triton_{test_pref}_o.bin")

if __name__ == "__main__":
    main()
import torch
import triton
import triton.language as tl
import math

from attn_utils import compute_relative_rmse, tensor_to_txt
from naive_reference import naive_forward_batched_supports, naive_backward_batched, flash2_backward_torch_dq, flash2_backward_torch_dkdv
from bwd_FA2_fp32 import flash2_bwd_wrapper as bwd_FA2_fp32_wrapper
from bwd_FA2_bf16 import flash2_bwd_wrapper as bwd_FA2_bf16_wrapper


def flash2_triton_bwd_eval(qkv, target_wrapper, target_dtype = torch.float32, name = "FA2"):
    q, k, v = qkv.unbind(dim=2)
    B, N, H, D = q.shape

    sm_scale = 1 / math.sqrt(D)
    S_naive_fwd, P_naive_fwd, O_naive_fwd, L_naive_fwd, l_naive_fwd, m_naive_fwd = naive_forward_batched_supports(q, k, v, sm_scale)
    dO_dummy = torch.randn_like(O_naive_fwd, device = qkv.device, dtype = qkv.dtype)



    dQ_ground, dK_ground, dV_ground, D_ground, dS_ground, dP_ground = naive_backward_batched(q, k, v, O_naive_fwd, dO_dummy, L_naive_fwd, sm_scale)

    """DEBUGGING KERNELS"""
    # dQ_ftorch_spec, dS_ftorch_spec = flash2_backward_torch_dq(q[0,:,0,:], k[0,:,0,:], v[0,:,0,:], O_naive_fwd[0,:,0,:], dO_dummy[0,:,0,:], L_naive_fwd[0,0,:], sm_scale)
    # dK_ftorch_spec, dV_ftorch_spec = flash2_backward_torch_dkdv(q[0,:,0,:], k[0,:,0,:], v[0,:,0,:], O_naive_fwd[0,:,0,:], dO_dummy[0,:,0,:], L_naive_fwd[0,0,:], sm_scale)
    # print("torch/torch_dQ RMSE: ", compute_relative_rmse(dQ_ftorch_spec, dQ_ground[0,:,0,:]))
    # print("torch/torch_dK RMSE: ", compute_relative_rmse(dK_ftorch_spec, dK_ground[0,:,0,:]))
    # print("torch/torch_dV RMSE: ", compute_relative_rmse(dV_ftorch_spec, dV_ground[0,:,0,:]))    


    """TRITON"""
    q = q.to(target_dtype)
    k = k.to(target_dtype)
    v = v.to(target_dtype)
    O_naive_fwd = O_naive_fwd.to(target_dtype)
    dO_dummy = dO_dummy.to(target_dtype)
    L_naive_fwd = L_naive_fwd.to(target_dtype)
    # sm_scale = sm_scale.to(target_dtype)

    print(f"q: {q.dtype}, k: {k.dtype}, v: {v.dtype}, O: {O_naive_fwd.dtype}, dO: {dO_dummy.dtype}, L: {L_naive_fwd.dtype}")
    D_flash2, dQ_flash2, dK_flash2, dV_flash2 = target_wrapper(q, k, v, O_naive_fwd, dO_dummy, L_naive_fwd, sm_scale)

    D_flash2 = D_flash2.to(torch.float32)
    dQ_flash2 = dQ_flash2.to(torch.float32)
    dK_flash2 = dK_flash2.to(torch.float32)
    dV_flash2 = dV_flash2.to(torch.float32)

    """RESULTS"""

    print(f"\nRESULTS FOR {name}")
    print("=====================================")
    print("triton / torch D RMSE: ", compute_relative_rmse(D_flash2, D_ground))
    print("triton / naive_dQ RMSE: ", compute_relative_rmse(dQ_flash2, dQ_ground))
    print("triton / naive_dK RMSE: ", compute_relative_rmse(dK_flash2, dK_ground))
    print("triton / naive_dV RMSE: ", compute_relative_rmse(dV_flash2, dV_ground))

    # print("triton / naive_dQ RMSE: ", compute_relative_rmse(dQ_flash2, dQ_ground, drop_dims=[0, 2]))
    # print("triton / naive_dK RMSE: ", compute_relative_rmse(dK_flash2, dK_ground, drop_dims=[0, 2]))
    # print("triton / naive_dV RMSE: ", compute_relative_rmse(dV_flash2, dV_ground, drop_dims=[0, 2]))


    # print("triton / naive_dQ ratio")
    # print(dQ_flash2[0,:, 0, :] / dQ_ground[0,:, 0, :])


    dQ_correct = torch.allclose(dQ_flash2, dQ_ground, rtol=5e-1, atol=1e-2)
    dK_correct = torch.allclose(dK_flash2, dK_ground, rtol=5e-1, atol=1e-2)
    dV_correct = torch.allclose(dV_flash2, dV_ground, rtol=5e-1, atol=1e-2)
    print(f"Correctness for {name}:")
    print(f"dQ_correct: {dQ_correct}")
    print(f"dK_correct: {dK_correct}")
    print(f"dV_correct: {dV_correct}")

    tensor_to_txt(dQ_ground[0, :, 0, :], output_name = f"logs/{name}_dQ")


"""
CUDA_VISIBLE_DEVICES=2 TRITON_PRINT_AUTOTUNING=1 python3 run_correctness.py
"""

if __name__ == "__main__":
    torch.manual_seed(0)


    batch_size = 1
    seq_len = 256
    n_heads = 32
    head_dim = 64
    qkv = torch.randn(batch_size, seq_len, 3, n_heads, head_dim, device = 'cuda', dtype=torch.float32)
    
    
    flash2_triton_bwd_eval(qkv, bwd_FA2_fp32_wrapper, name = "Flash2 for FP32")
    flash2_triton_bwd_eval(qkv, bwd_FA2_bf16_wrapper, torch.bfloat16, name = "Flash2 for BF16")


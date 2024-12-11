import torch
from torch.autograd import gradcheck
# from naive_wrapped import *
from naive_mha_wrapped import naive_AttnQKVPackedFunc
from triton_FWDfp32_wrapped import triton_AttnQKVPackedFunc

def test_custom_attn(attn_func):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 2
    seq_len = 32
    n_heads = 2
    head_dim = 16  # Must be divisible by 8 as per assertion
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qkv = torch.randn(batch_size, seq_len, 3, n_heads, head_dim, 
                      dtype=torch.float64,  # Use double for better numerical precision
                        device=device,
                      requires_grad=True)
    
    attn_func = attn_func.apply
    
    # Test backward pass with gradcheck
    try:
        test_input = qkv.clone().detach().requires_grad_(True)
        print("Running gradient check...")
        result = gradcheck(attn_func, (test_input,), 
                         eps=1e-6, 
                         atol=1e-4,
                         check_backward_ad=True)
        assert result
        print("✓ Gradient check passed")
    except Exception as e:
        print(f"✗ Gradient check failed: {str(e)}")
        raise

    
    print("All tests passed successfully!")

if __name__ == "__main__":
    # test_custom_attn(naive_AttnQKVPackedFunc)
    test_custom_attn(triton_AttnQKVPackedFunc)
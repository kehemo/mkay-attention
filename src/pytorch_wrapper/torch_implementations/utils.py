import torch

def check_tensor(name, expected, actual):
    try:
        assert torch.allclose(expected, actual, rtol = 1e-5, atol = 5e-3), "Tensor wrong"
        print(f"{name} is correct")
    except:
        print(f"\n================")
        print(f"Expected for {name}")
        print(expected)
        print("Actual")
        print(actual)
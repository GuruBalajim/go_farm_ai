import torch

# Assuming tensor is the one causing the issue
with torch.inference_mode():
    tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)

# Outside inference mode
tensor_clone = tensor.clone()  # Create a regular tensor
tensor_clone[0] = 10.0  # Now you can safely perform in-place operations
print(tensor_clone)

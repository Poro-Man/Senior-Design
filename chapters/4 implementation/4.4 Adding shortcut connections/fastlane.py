import torch
import torch.nn as nn

class ExampleDeepNeuralNetwork(nn.Module):
    """
    A simple feed-forward network that can add shortcut (residual) connections
    when the input and output of a layer have the same shape.
    - layer_sizes: list[int], e.g., [1, 64, 64, 64, 64, 1]
    - use_shortcut: bool, whether to attempt residual adds when shapes match
    """
    def __init__(self, layer_sizes, use_shortcut: bool = True):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.GELU())
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and (x.shape == layer_output.shape):
                x = x + layer_output
            else:
                x = layer_output
        return x

def print_gradients(model: nn.Module, x: torch.Tensor):
    model.zero_grad(set_to_none=True)
    output = model(x)
    target = torch.tensor([[0.0]], dtype=output.dtype, device=output.device)
    loss = nn.MSELoss()(output, target)clear

    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item():.6e}")

if __name__ == "__main__":
    torch.manual_seed(42)
    layer_sizes = [1, 64, 64, 64, 64, 1]  # inner 64→64 layers enable residual adds
    x = torch.randn(1, 1)

    print("=== Gradients WITHOUT shortcuts ===")
    model_plain = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print_gradients(model_plain, x)

    print("\n=== Gradients WITH shortcuts ===")
    model_res = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_res, x)

import torch
import torch.nn as nn
import tiktoken

#################################################
#
#
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
#################################################
#
#
class ExampleDNN(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut

        self.layers = nn.ModuleList([
            nn.Sequential( nn.Linear(layer_sizes[0], layer_sizes[1]), GELU() ),
            nn.Sequential( nn.Linear(layer_sizes[1], layer_sizes[2]), GELU() ),
            nn.Sequential( nn.Linear(layer_sizes[2], layer_sizes[3]), GELU() ),
            nn.Sequential( nn.Linear(layer_sizes[3], layer_sizes[4]), GELU() ),
            nn.Sequential( nn.Linear(layer_sizes[4], layer_sizes[5]), GELU() )            
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)

            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        
        return x

#################################################
#
#    
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

#################################################
#
#    
if __name__ == "__main__":
    layer_sizes = [3,3,3,3,3,1]

    sample_input = torch.tensor([[1., 0., -1.]])

    print(sample_input)

    print("------------------------------------")
    print("-----------without shortcut---------")
    print("------------------------------------")
    
    torch.manual_seed(123)
    model_without_shortcut = ExampleDNN(
        layer_sizes, use_shortcut=False
    )
    print_gradients(model_without_shortcut, sample_input)

    print("------------------------------------")    
    print("-----------with shortcut------------")
    print("------------------------------------")
    
    torch.manual_seed(123)
    model_without_shortcut = ExampleDNN(
        layer_sizes, use_shortcut=True
    )
    print_gradients(model_without_shortcut, sample_input)    

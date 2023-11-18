import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Simplenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        #print(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    model = Simplenet()
    print(model)
    
    X = torch.rand(1, 28, 28)
    #print(X)
    
    log = model(X)
    print(log)
    
    pred = nn.Softmax(dim=1)(log)
    y_pred = pred.argmax(1)
    print(pred)
    print(y_pred)
    
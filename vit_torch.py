import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import matplotlib.pyplot as plt

from einops import repeat
from einops.layers.torch import Rearrange

###############################################
#
class Patching(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = patch_size, pw = patch_size)
        
    def forward(self, x):
        x = self.net(x)
        return x
                    
class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)
    
    def forward(self, x):
        x = self.net(x)
        return x
   
class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches+1, dim))
    
    def forward(self, x):
        batch_size, _, __ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b = batch_size)
        x = torch.concat([cls_tokens, x], dim = 1)
        x += self.pos_embedding
        return x
        
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim
        
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        
        self.split_into_heads = Rearrange("b n (h d) ->b h n d", h = self.n_heads)
        self.softmax = nn.Softmax(dim = -1)
        self.concat = Rearrange("b h n d -> b n (h d)", h=self.n_heads)
        
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)
        
        logit = torch.matmul(q, k.transpose(-1,-2)) * (self.dim_heads ** -0.5)
        attention_weight = self.softmax(logit)
        
        output = torch.matmul(attention_weight, v)
        output = self.concat(output)
        return output                
    
class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.multi_head_attention = MultiHeadAttention(dim=dim, n_heads=n_heads)
        self.mlp = MLP(dim=dim, hidden_dim = mlp_dim)
        self.depth = depth
        
    def forward(self, x):
        for _ in range(self.depth):
            x = self.multi_head_attention(self.norm(x)) + x
            x = self.mlp(self.norm(x)) + x
            
        return x
            
class MLPHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_classes, dim, depth, n_heads, channels = 3, mlp_dim = 256):

        super().__init__()
        
        # Params
        n_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size
        self.depth = depth

        # Layers
        self.patching = Patching(patch_size = patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim = patch_dim, dim = dim)
        self.embedding = Embedding(dim = dim, n_patches = n_patches)
        self.transformer_encoder = TransformerEncoder(dim = dim, n_heads = n_heads, mlp_dim = mlp_dim, depth = depth)
        self.mlp_head = MLPHead(dim = dim, out_dim = n_classes)


    def forward(self, img):

        x = img

        #print(x.shape)
        x = self.patching(x)

        x = self.linear_projection_of_flattened_patches(x)

        x = self.embedding(x)

        x = self.transformer_encoder(x)

        x = x[:, 0]
        x = self.mlp_head(x)

        return x

###############################################
#
def build_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    image_size = 32
    patch_size = 4
    n_classes = 10
    dim = 256
    depth = 3
    n_heads = 4

    model = ViT(image_size, patch_size, n_classes, dim, depth, n_heads).to(device)
  
    return model, device    

def load_cifare10():
    transform = transforms.Compose(    [transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    batch_size = 100   
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')   

    return train_loader, test_loader, classes
    
def test_model():        
    model, device = build_model()    
  
    image_size = 32
    patch_size = 4
    depth = 3

    X = torch.rand(patch_size, depth, image_size, image_size)   

    x = model(X)
        
    print(x)
   
def train_model():
    net, device = build_model()    
    print(net)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    train_loader, test_loader, classes = load_cifare10()

    epochs = 10
    for epoch in range(0,epochs):
        epoch_train_loss    = 0
        epoch_train_acc     = 0
        epoch_test_loss     = 0
        epoch_test_acc      = 0
        
        #########################
        #
        net.train()
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()/len(train_loader)
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_train_acc += acc/len(train_loader)
            
            del inputs
            del outputs
            del loss
            
        #########################
        #
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                
                epoch_test_loss += loss.item()/len(test_loader)
                test_acc = (outputs.argmax(dim=1) == labels).float().mean()
                epoch_test_acc += test_acc/len(test_loader)                
        
        print(f"Epoch {epoch+1} : train acc={epoch_train_acc:.2f} train loss={epoch_train_loss:.2f}") 
        print(f"Epoch {epoch+1} : test acc={epoch_test_acc:.2f} test loss={epoch_test_loss:.2f}") 
        
###############################################
#
if __name__ == '__main__':
#    test_model()
    train_model()
    
    
    


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import datasets

#from torchvision.transforms import ToTensor, Lambda

transform = transforms.ToTensor()

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1) )
)

fig, axs = plt.subplots(4, 4, figsize=(8,8) )

for i in range(4):
  for j in range(4):
    image, label = ds[i*4 + j]
    image_numpy = image.numpy().squeeze()
    axs[i,j].imshow(image_numpy, cmap="gray")
    axs[i,j].axis("off")
    axs[i,j].set_title(f"Label: {label}")

plt.tight_layout()
plt.show()

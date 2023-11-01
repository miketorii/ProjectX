import torch
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def displayimg(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow( np.transpose(npimg, (1,2,0)) )
    plt.show()
    
def main():
    transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ])
    
    trainset = tv.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=1)
    
    dataiter = iter(trainloader)
    #imgs, lbls = dataiter.next()
    imgs, lbls = next(dataiter)
    
    displayimg(tv.utils.make_grid(imgs[0]))
    

if __name__ == "__main__":
    main()
    

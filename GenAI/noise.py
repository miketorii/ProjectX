import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

def reverse_to_img(x):
    x = x*255
    x = x.clamp(0,255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)

def showimgs(imgs):
    plt.figure(figsize=(15,6))
    for i, img in enumerate(imgs[:10]):
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        plt.title(f"Noise: {i+100}")
        plt.axis("off")
    plt.show()

def add_noise(x_0, t, betas):
    T = len(betas)
    assert t >= 1 and t <= T
    t_idx = t-1

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bar = alpha_bars[t_idx]

    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar)*x_0 + torch.sqrt(1-alpha_bar)*eps

    return x_t

if __name__=="__main__":
    print("---start---")
    '''
    x = torch.randn(3,64,64)
    T = 1000
    betas = torch.linspace(0.01, 0.02, T)

    for t in range(T):
        beta = betas[t]
        eps = torch.randn_like(x)
        x = torch.sqrt( (1-beta) * x + torch.sqrt(beta) * eps )
    '''

    image = plt.imread("flower.png")
    print(image.shape)

    preprocess = transforms.ToTensor()
    x = preprocess(image)
    print(x.shape)

    original_x = x.clone()

    ###################
    T = 1000
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, T)
    imgs = []

    for t in range(T):
        if t%100 == 0:
            img = reverse_to_img(x)
            imgs.append(img)
        
        beta = betas[t]
        eps = torch.randn_like(x)
        x = torch.sqrt( (1-beta) * x + torch.sqrt(beta) * eps )

    showimgs(imgs)

    x = original_x

    t = 100
    x_t = add_noise(x, t, betas)

    img = reverse_to_img(x_t)
    plt.imshow(img)
    plt.title(f"Noise: {t}")
    plt.axis("off")
    plt.show()
    
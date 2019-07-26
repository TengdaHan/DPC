# utilities for visualization
import torch
import numpy as np
import torchvision
from torchvision import transforms

def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean)==len(std)==3
    inv_mean = [-mean[i]/std[i] for i in range(3)]
    inv_std = [1/i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)

def vis_input(tensor, nrow):
    if tensor.is_cuda: tensor = tensor.cpu()
    denorm_transform = denorm()
    to_pil = transforms.ToPILImage()
    return to_pil(denorm_transform(torchvision.utils.make_grid(tensor, nrow=nrow)))

def vis_gt(tensor, nrow):
    if tensor.is_cuda: tensor = tensor.cpu()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)

    to_pil = transforms.ToPILImage()

    return to_pil(torchvision.utils.make_grid(tensor, nrow=nrow).float())

def vis_output(tensor, nrow):
    if tensor.is_cuda: tensor = tensor.cpu()
    assert tensor.size(1) == 2
    _, idx = torch.max(tensor, 1)
    idx = idx.unsqueeze(1)

    to_pil = transforms.ToPILImage()

    return to_pil(torchvision.utils.make_grid(idx, nrow=nrow).float())

def vis_filter(tensor, nrow):
    channel = 3
    assert tensor.dim() == 4
    if tensor.is_cuda: tensor = tensor.cpu()
    if tensor.size(1) == 2:
        channel = 2
        tensor = torch.cat((tensor, torch.zeros(tensor.size(0), 1, tensor.size(2), tensor.size(3))), 1)
    assert tensor.size(1) == 3
    array = torchvision.utils.make_grid(tensor, nrow).numpy()
    if channel == 2: array = array[:,:,0:2]
    return array

def plot_kernels(tensor, num_cols=8,name='none.png'):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.imsave(name)

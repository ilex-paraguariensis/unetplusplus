import torch

pretrained_net = torch.hub.load(
    "milesial/Pytorch-UNet", "unet_carvana", pretrained=True, scale=0.5
)

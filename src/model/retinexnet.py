# retinexnet complete model combining decomnet and relightnet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .decomnet import DecomNet
from .relightnet import RelightNet


class RetinexNet(nn.Module):
    # full retinexnet model with loss computation for training
    
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()

    def forward(self, input_low, input_high):
        # handle numpy array inputs
        if isinstance(input_low, np.ndarray):
            input_low = torch.from_numpy(input_low).float()
        if isinstance(input_high, np.ndarray):
            input_high = torch.from_numpy(input_high).float()
        
        # move to correct device
        model_device = next(self.parameters()).device
        if input_low.device != model_device:
            input_low = input_low.to(model_device)
        if input_high.device != model_device:
            input_high = input_high.to(model_device)
            
        # decompose both low and high light images
        R_low, I_low = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)

        # enhance illumination
        I_delta = self.RelightNet(I_low, R_low)

        # expand illumination to 3 channels for reconstruction
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

        # compute losses
        self.recon_loss_low = F.l1_loss(R_low * I_low_3, input_low)
        self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)
        self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, input_low)
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)
        self.equal_R_loss = F.l1_loss(R_low, R_high.detach())
        self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)

        # smoothness losses
        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        # total decomposition loss
        self.loss_Decom = (self.recon_loss_low + self.recon_loss_high +
                          0.001 * self.recon_loss_mutal_low +
                          0.001 * self.recon_loss_mutal_high +
                          0.1 * self.Ismooth_loss_low +
                          0.1 * self.Ismooth_loss_high +
                          0.01 * self.equal_R_loss)
        
        # relighting loss
        self.loss_Relight = self.relight_loss + 3 * self.Ismooth_loss_delta

        # store outputs for visualization
        self.output_R_low = R_low.detach().cpu()
        self.output_I_low = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S = R_low.detach().cpu() * I_delta_3.detach().cpu()
        
        return self.loss_Decom + self.loss_Relight
    
    def inference(self, input_low):
        # inference mode for testing
        if isinstance(input_low, np.ndarray):
            input_low = torch.from_numpy(input_low).float()
        
        model_device = next(self.parameters()).device
        if input_low.device != model_device:
            input_low = input_low.to(model_device)
        
        with torch.no_grad():
            R_low, I_low = self.DecomNet(input_low)
            I_delta = self.RelightNet(I_low, R_low)
            I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)
            output = R_low * I_delta_3
        
        return R_low, I_low, I_delta, output

    def gradient(self, input_tensor, direction):
        # compute image gradients
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
            
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        # smoothness loss weighted by reflectance edges
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        
        return torch.mean(
            self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
            self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y"))
        )

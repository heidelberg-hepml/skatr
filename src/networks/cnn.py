import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint

class CNN(nn.Module):
    
    def __init__(self, cfg:DictConfig):
        super().__init__()

        self.cfg = cfg
        in_ch, hi_ch, ou_ch = cfg.in_channels, cfg.hidden_channels, cfg.out_channels
        self.conv1 = nn.Conv3d(in_ch, hi_ch, kernel_size=(3,3,102), bias=True, stride=(1,1,102))
        self.conv2 = nn.Conv3d(hi_ch, hi_ch, kernel_size=(3,3,2), bias=True)
        self.conv3 = nn.Conv3d(hi_ch, 2*hi_ch, kernel_size=(3,3,2), bias=True)
        self.conv3_zero = nn.Conv3d(2*hi_ch, 2*hi_ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
        self.conv4 = nn.Conv3d(2*hi_ch, 4*hi_ch, kernel_size=(3,3,2), bias=True)
        self.conv4_zero = nn.Conv3d(4*hi_ch, 4*hi_ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
        
        act = nn.ReLU()
        max = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        avg = nn.AvgPool3d(kernel_size=(31,31,18))
        self.act = act if not cfg.checkpoint_grads else lambda x: checkpoint(act, x)
        self.max = max if not cfg.checkpoint_grads else lambda x: checkpoint(max, x)
        self.avg = avg if not cfg.checkpoint_grads else lambda x: checkpoint(avg, x)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,128, bias=True)
        self.linear2 = nn.Linear(128,128, bias=True)
        self.linear3 = nn.Linear(128,128, bias=True)
        self.out = nn.Linear(128, ou_ch, bias=True)
        self.out_act = getattr(F, cfg.out_act) if cfg.out_act else None
    
    def forward(self,x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.max(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv3_zero(x))
        x = self.max(x)
        x = self.act(self.conv4(x))
        x = self.act(self.conv4_zero(x))
        x = self.avg(x)
        x = self.flatten(x)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.act(self.linear3(x))
        x = self.out(x)
        if self.cfg.out_act:
            x = self.out_act(x)
        return x
    
class MediumCNN(nn.Module):
    
    def __init__(self, cfg:DictConfig):
        super().__init__()

        self.cfg = cfg
        in_ch, hi_ch, ou_ch = cfg.in_channels, cfg.hidden_channels, cfg.out_channels
        self.conv1 = nn.Conv3d(in_ch, hi_ch, kernel_size=(3,3,51), bias=True, stride=(1,1,51))
        self.conv2 = nn.Conv3d(hi_ch, hi_ch, kernel_size=(3,3,2), bias=True)
        self.conv3 = nn.Conv3d(hi_ch, 2*hi_ch, kernel_size=(3,3,2), bias=True)
        self.conv3_zero = nn.Conv3d(2*hi_ch, 2*hi_ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
        self.conv4 = nn.Conv3d(2*hi_ch, 4*hi_ch, kernel_size=(3,3,2), bias=True)
        self.conv4_zero = nn.Conv3d(4*hi_ch, 4*hi_ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
        
        act = nn.ReLU()
        max = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        avg = nn.AvgPool3d(kernel_size=(13,13,18))
        self.act = act if not cfg.checkpoint_grads else lambda x: checkpoint(act, x)
        self.max = max if not cfg.checkpoint_grads else lambda x: checkpoint(max, x)
        self.avg = avg if not cfg.checkpoint_grads else lambda x: checkpoint(avg, x)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,128, bias=True)
        self.linear2 = nn.Linear(128,128, bias=True)
        self.linear3 = nn.Linear(128,128, bias=True)
        self.out = nn.Linear(128, ou_ch, bias=True)
        self.out_act = getattr(F, cfg.out_act) if cfg.out_act else None
    
    def forward(self,x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.max(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv3_zero(x))
        x = self.max(x)
        x = self.act(self.conv4(x))
        x = self.act(self.conv4_zero(x))
        x = self.avg(x)
        x = self.flatten(x)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.act(self.linear3(x))
        x = self.out(x)
        if self.cfg.out_act:
            x = self.out_act(x)
        return x    

# class ConvNet3D_downsized(nn.Module):
    
#     def __init__(self, in_ch=1, ch=32, N_parameter=6,sigmoid=False):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_hi_ch, hi_ch, kernel_size=(3,3,51), bias=True, stride=(1,1,51))
#         self.conv2 = nn.Conv3d(hi_ch, hi_ch, kernel_size=(3,3,2), bias=True)
#         self.conv3 = nn.Conv3d(hi_ch, 2*hi_ch, kernel_size=(3,3,2), bias=True)
#         self.conv3_zero = nn.Conv3d(2*hi_ch, 2*hi_ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
#         self.conv4 = nn.Conv3d(2*hi_ch, 4*hi_ch, kernel_size=(3,3,2), bias=True)
#         self.conv4_zero = nn.Conv3d(4*hi_ch, 4*hi_ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
#         self.max = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
#         self.avg = nn.AvgPool3d(kernel_size = (13,13,18))
#         self.flatten = nn.Flatten()
#         self.linear1 = nn.Linear(128,128, bias=True)
#         self.linear2 = nn.Linear(128,128, bias=True)
#         self.linear3 = nn.Linear(128,128, bias=True)
#         self.out = nn.Linear(128, N_parameter, bias=True)
#         self.sigmoid = sigmoid
    
#     def forward(self,x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.max(x)
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv3_zero(x))
#         x = self.max(x)
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv4_zero(x))
#         x = self.avg(x)
#         x = self.flatten(x)
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear3(x))
#         if self.sigmoid:
#             x = F.sigmoid(x)

#         return self.out(x)

def subnet_fc(dims_in,dims_out):
    return nn.Sequential(nn.Linear(dims_in,256),nn.ReLU(),
            nn.Linear(256,dims_out))

def load_model(location,N_parameter,cINN=False,downsize=False,sigmoid=False):
    if cINN:
        N_DIM=N_parameter
        cond_dims=(N_parameter,)
        model = Ff.SequenceINN(N_DIM)
        for k in range(8):
            model.append(Fm.AllInOneBlock, cond=0,cond_shape=cond_dims,subnet_constructor=subnet_fc)
    elif downsize:
        model = ConvNet3D_downsized(N_parameter = N_parameter,sigmoid=sigmoid)
    else:
        model = ConvNet3D(N_parameter = N_parameter)
    model.load_state_dict(torch.load(location))
    model.eval()
    return model

'''
#if you want to know the number of parameters
model1 = ConvNet3D()
num_params = sum(p.numel() for p in model1.parameters())
print(f"Number of parameters in the model: {num_params}")

'''
"""
## Adapt from MPRNet code
## Original header:
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from MPRNet import MPRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx
import pickle as pkl
from KAIRmaster.utils.utils_modelsummary import get_model_activation 
from KAIRmaster.utils.utils_modelsummary import get_model_flops
import sys

train_size=(1,3,256,256)

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )
           
    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]
            
            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        if self.fast_imp:   # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                out = F.adaptive_avg_pool2d(x,1)
            else:
                r1 = [r for r in self.rs if h%r==0][0]
                r2 = [r for r in self.rs if w%r==0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1,0,1,0)) # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)
    
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        
        return out

def replace_layers(model, base_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, fast_imp, **kwargs)
            
        if isinstance(m, nn.AdaptiveAvgPool2d): 
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, **kwargs)
            assert m.output_size == 1
            setattr(model, n, pool)

        if isinstance(m, nn.InstanceNorm2d):
            # Not implemented
            assert 0

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
device_ids = [i for i in range(torch.cuda.device_count())]
model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)
model_restoration.eval()

# TLC operations
# replace_layers(model_restoration, base_size=384, fast_imp=False, auto_pad=False)
# imgs = torch.rand(train_size)
# with torch.no_grad():
#     model_restoration.forward(imgs, torch.tensor([7]))

dataset = args.dataset
if True:
    windowSizes = [7,9,11,13,15,17,19,27,31]
    # windowSizes = [-1] # -1 stands for inference
    for window in windowSizes:
        rgb_dir_test = "Datasets/test/input%d"%(window)
        test_dataset = get_test_data(rgb_dir_test, img_options={})
        test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
        result_dir = "results/GoPro_rev2_6cab_again_%d"%(window)
        utils.mkdir(result_dir)

        with torch.no_grad():
            if window > 17:
                # for out of distribution images, extrapolate with only 1 extra CAB
                window_tensor = torch.tensor([17] * 2)
            else:
                window_tensor = torch.tensor([window] * 2)

            for data_test in tqdm(test_loader):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                input_    = data_test[0].cuda()
                filenames = data_test[1]

                # Padding in case images are not multiples of 8
                if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
                    factor = 8
                    h,w = input_.shape[2], input_.shape[3]
                    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                    padh = H-h if h%factor!=0 else 0
                    padw = W-w if w%factor!=0 else 0
                    input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
                window_tensor = window_tensor.to("cuda")

                restored = model_restoration(input_, window_tensor)
                output_windowsize = restored[1]
                window_diffs = restored[2]
                restored = torch.clamp(restored[0],0,1)

                # Unpad images to original dimensions
                if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
                    restored = restored[:,:,:h,:w]

                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                for batch in range(len(restored)):
                    restored_img = img_as_ubyte(restored[batch])
                    utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)


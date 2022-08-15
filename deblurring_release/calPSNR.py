# from asyncio import windows_utils
import os
import numpy as np
from skimage import io
import torch
from scipy import signal
from math import exp

import tqdm

import torch.nn.functional as F
from torch.autograd import Variable


def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


datasets = ['GoPro']
windowsizes = [7,9,11,13,15,17,19,23,27,31]
for window in windowsizes:
    print("Current: %d"%(window))
    file_path = "./results/GoPro_window%d/"%(window)
    gt_path = "./Datasets/test/target%d/"%(window)
    
    path_list = sorted(os.listdir(file_path))
    gt_list = sorted(os.listdir(gt_path))

    img_num = len(path_list)
    total_psnr = 0
    for i in tqdm.tqdm(range(img_num)):
        input = io.imread(file_path + path_list[i])
        gt = io.imread(gt_path + gt_list[i])
        total_psnr += numpyPSNR(input, gt)

    total_psnr /= len(path_list)
    print(total_psnr)


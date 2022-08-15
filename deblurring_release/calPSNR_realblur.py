# from asyncio import windows_utils
import os
import numpy as np
from skimage import io
import torch
import cv2
from scipy import signal
from math import exp

import tqdm
import gauss

import torch.nn.functional as F
from torch.autograd import Variable

def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x.astype('float32'), cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs.astype('float32'), cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

def numpyPSNR_realblur(tar_img, prd_img):
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0

    image_test, image_true, cr1, shift = image_align(prd_img, tar_img)

    image_mask = cr1
    data_range = 1
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10((data_range ** 2) / err)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

path_list = []
gt_list = []
file_path_root = "Datasets/"
with open(file_path_root + "RealBlur_R_test_list.txt", "r") as f:
    contents = f.read()
    rows = contents.split('\n')
for i in range(len(rows)-1):
    rows[i] = rows[i][:-1]
for i in range(len(rows)):
    gt_path, blur_path = rows[i].split()
    scene_index = int(gt_path.split('/')[1][5:])
    deblur_path = "results/RealBlur_R/scene%d/"%(scene_index) + blur_path.split('/')[-1]
    gt_path = file_path_root + gt_path
    path_list.append(deblur_path)
    gt_list.append(gt_path)

img_num = len(path_list)
total_psnr = 0
total_ssim = 0
for i in tqdm.tqdm(range(img_num)):
    input = io.imread(path_list[i])
    gt = io.imread(gt_list[i])
    total_psnr += numpyPSNR_realblur(gt, input)
total_psnr /= len(path_list)
print(total_psnr)

import cv2
import numpy as np
import os

for mode in ["train", "test"]:
  files = os.listdir("Datasets/GoPro_Large_all/" + mode)
  if mode == "test": 
    windows = [7,9,11,13,15,17,19,23,27,31]
    for window in windows:
      # Different from training, testing blurs with different blur level are separated for evaluation
      os.makedirs('Datasets/test/input%d'%(window), exist_ok=True)
      os.makedirs('Datasets/test/target%d'%(window), exist_ok=True)
      for each_video in files:
        print("Creating for video %s with size %s"%(each_video, window))
        images = os.listdir("Datasets/GoPro_Large_all/%s/"%(mode) + each_video)
        images.sort()
        count = 0
        h = 720
        w = 1280
        arrayOfarr = []
        for i, each_image in enumerate(images):
          readImage = cv2.imread("Datasets/GoPro_Large_all/%s/"%(mode) + each_video + "/" + each_image)
          count += 1
          arrayOfarr.append(readImage)
          if count > window:
            arrayOfarr.pop(0)
          if (count >= window) and ((count - window) % window == 0) and len(arrayOfarr) == window:
            arr = np.zeros((h, w, 3), float)
            for each in arrayOfarr:
              arr += each/window
            cv2.imwrite('Datasets/%s/input%d/%s_test%s.png'%(mode, window, each_video, (count - window)//window), arr)
            cv2.imwrite('Datasets/%s/target%d/%s_test%s.png'%(mode, window, each_video, (count - window)//window), arrayOfarr[window//2])
  else:
    windows = [7,9,11,13,15]
    # original training files are not used for training.
    os.rename('Datasets/train/input', 'Datasets/train/input_original')
    os.rename('Datasets/train/target', 'Datasets/train/target_original')
    os.makedirs('Datasets/train/input', exist_ok=True)
    os.makedirs('Datasets/train/target', exist_ok=True)
    for window in windows:
      for each_video in files:
        print("Creating for video %s with size %s"%(each_video, window))
        images = os.listdir("Datasets/GoPro_Large_all/%s/"%(mode) + each_video)
        images.sort()
        count = 0
        h = 720
        w = 1280
        arrayOfarr = []
        for i, each_image in enumerate(images):
          readImage = cv2.imread("Datasets/GoPro_Large_all/%s/"%(mode) + each_video + "/" + each_image)
          count += 1
          arrayOfarr.append(readImage)
          if count > window:
            arrayOfarr.pop(0)
          if (count >= window) and ((count - window) % window == 0) and len(arrayOfarr) == window:
            arr = np.zeros((h, w, 3), float)
            for each in arrayOfarr:
              arr += each/window
            cv2.imwrite('Datasets/%s/input/%s_test%s_w%s.png'%(mode, each_video, (count - window)//window, window), arr)
            cv2.imwrite('Datasets/%s/target/%s_test%s_w%s.png'%(mode, each_video, (count - window)//window, window), arrayOfarr[window//2])
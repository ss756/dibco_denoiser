

import torch

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os

def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img
def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
  fig = plt.figure(figsize=(25, 10))
  ax1 = fig.add_subplot(1, 2, 1) 
  plt.title('Input image', fontsize=16)
  ax1.axis('off')
  ax2 = fig.add_subplot(1, 2, 2)
  plt.title('NAFNet output', fontsize=16)
  ax2.axis('off')
  ax1.imshow(img1)
  ax2.imshow(img2)

def single_image_inference(model, img, save_path):
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})
    
    if model.opt['val'].get('grids', False):
        model.grids()
    
    model.test()
    
    if model.opt['val'].get('grids', False):
        model.grids_inverse()
    
    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    # imwrite(sr_img, save_path)

    return sr_img

opt_path = 'options/test/SHABBY/shabby_x4.yml'
opt = parse(opt_path, is_train=False)
opt['dist'] = False
NAFNet = create_model(opt)

# %%
input_path = 'datasets/shabby/test/noisy/'
output_path = 'datasets/shabby/test/clean_predicted/'

all_file_path = glob(input_path+'*.png')

for i, single_path in enumerate(all_file_path):
    
    print('img_'+str(i+1)+' out of total images '+str(len(all_file_path)))


    img_input = imread(single_path)
    inp = img2tensor(img_input)
    img_output = single_image_inference(NAFNet, inp, output_path)

    
    output_single_path = output_path+os.path.basename(single_path)
    cv2.imwrite(output_single_path, img_output)
    # display(img_input, img_output)



# %%






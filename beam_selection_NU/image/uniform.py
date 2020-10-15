from __future__ import print_function

import numpy as np
import sys
import os
import glob
import random

from PIL import Image
from random import randrange
from tqdm import tqdm

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator



def show_all_files_in_directory(input_path):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".png"):
               files_list.append(os.path.join(path, file))
    return files_list


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False



background = show_all_files_in_directory('/home/batool/beam_selection/image/crops/background')
car = show_all_files_in_directory('/home/batool/beam_selection/image/crops/car')
bus = show_all_files_in_directory('/home/batool/beam_selection/image/crops/bus')
truck = show_all_files_in_directory('/home/batool/beam_selection/image/crops/truck')
save_path = '/home/batool/beam_selection/image/crops/aug/car/'


short_list = random.sample(car, len(truck)-len(car))
#short_list = random.sample(car, 9601-1)

count = len(car)

for i in tqdm(short_list):
    img=load_img(i)
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(brightness_range=[0.5,1.5])
    it = datagen.flow(samples, batch_size=1)


    for r in range(1):
        batch = it.next()
        image = batch[0].astype('uint8')
        img = Image.fromarray(image,mode='RGB')
        img.save(save_path+str(count)+'.png')
        count+=1
print(count)


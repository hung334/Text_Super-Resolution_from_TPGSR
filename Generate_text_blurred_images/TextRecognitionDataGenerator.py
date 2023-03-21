# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:49:36 2022

@author: chen_hung
"""
from PIL import Image
import matplotlib.pyplot as plt
import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image
from trdg.generators import GeneratorFromStrings
import random
from random import choice
import time
import os

#random.randint(1, 6)

#font_format = './Simplified_Chinese/MFBanHei_Noncommercial-Regular.otf'



def Generator_str_to_img(input_string,font_format=[],stroke_w=1):
        
    # The generators use the same arguments as the CLI, only as parameters
    HR_generator = GeneratorFromStrings(
        [input_string],
        fonts = font_format,
        blur=0,random_blur=False,
        language = "cn",#語言設定
        size = 64,#文字大小
        width = -1,#生成圖片寬度
        count = 1,#生成數量
        background_type = 2,#背景選擇，0:gaussian_noise、1:plain_white、2:quasicrystal、3:image
        stroke_width = stroke_w, #文字筆劃的寬度
        distorsion_type = 0, #文字失真  0:None,1:sin,2:cos,other:random
    )
   
    return next(HR_generator)[0]

def Diversity_generator_str_to_img(input_string,fonts_path,fonts_datas):
    
    #font_format = []
    if(random.random()>0.5):
         font_format=[os.path.join(fonts_path,choice(fonts_datas))]
    else:
         font_format = []
    
    # The generators use the same arguments as the CLI, only as parameters
    HR_generator = GeneratorFromStrings(
        [input_string],
        fonts = font_format,
        blur=0,random_blur=False,
        language = "cn",#語言設定
        size = 64,#文字大小
        width = -1,#生成圖片寬度
        count = 1,#生成數量
        background_type = 1,#random.randint(0, 2),#背景選擇，0:gaussian_noise、1:plain_white、2:quasicrystal、3:image
        stroke_width = random.randint(1, 3), #文字筆劃的寬度
        distorsion_type = 0, #文字失真  0:None,1:sin,2:cos,other:random
    )
   
    return next(HR_generator)[0]

if __name__ == '__main__':
    
    
    fonts_path = './Simplified_Chinese'
    fonts_datas = os.listdir(fonts_path)
    
    
    t1 = time.time()
    for i in os.listdir('./Simplified_Chinese'):
        input_string ='上海证券交易所网站' #'上海证券交易所网站'
        #HR_img = Generator_str_to_img(input_string,stroke_w=3,font_format=[os.path.join(fonts_path,i)])
        HR_img = Diversity_generator_str_to_img(input_string,fonts_path,fonts_datas)
        plt.imshow(HR_img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    #print(time.time()-t1)
    
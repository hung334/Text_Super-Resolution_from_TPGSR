# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 17:14:08 2023

@author: chen_hung
"""

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageFilter, ImageStat
from Generate_Stamp.generatepic_simple_stamp import generate_stamp
from Generate_Stamp.generate_complex_stamp import Stamp


def transparent(image,pha=200):
    image = image.convert('RGBA')
    image.putalpha(pha)#0~255,255最深，0最淺
    #print(image.mode)
    # Transparency
    newImage = []
    for item in image.getdata():
        if item[:3] == (255, 255, 255):
            newImage.append((255, 255, 255, 0))
        else:
            newImage.append(item)
    
    image.putdata(newImage)
    return image

def random_stamp(image):
    # (left, upper, right, lower)
    w,h = image.size
    try:
        return image.crop((random.randint(0,w),random.randint(0,h),random.randint(0,w),random.randint(0,h)))
    except :
        return image.crop((random.randint(0,w//2),random.randint(0,h//2),random.randint(w//2,w),random.randint(h//2,h)))

def random_crop_stamp(stamp,text_img_size):
    # (left, upper, right, lower)
    s_w,s_h = stamp.size
    t_w,t_h = text_img_size
    #try:
    new_x = random.randint(0,s_w-t_w)
    new_y = random.randint(0,s_h-t_h)
    return stamp.crop((new_x,new_y,new_x+t_w,new_y+t_h))
    #except :
    #    return image.crop((random.randint(0,w//2),random.randint(0,h//2),random.randint(w//2,w),random.randint(h//2,h)))
    #    print('error')

#基于gbk2312码生成的汉字，大概有6千个常用的汉字，所以生成的汉字，我们大多都认识
def GBK2312():
    head = random.randint(0xb0, 0xf7)
    body = random.randint(0xa1, 0xf9)   # 在head区号为55的那一块最后5个汉字是乱码,为了方便缩减下范围
    val = f'{head:x}{body:x}'
    str = bytes.fromhex(val).decode('gb2312')
    return str


def get_num(n=4):
    a=''
    for i in range(n):
        a+=str(random.randint(0,9))
    return a

def get_str(n=4):
    a=''
    for i in range(n):
        a+=GBK2312()
    return a

def blur_img(img,blur = random.randint(8,10)):
        gaussian_filter = ImageFilter.GaussianBlur(radius=blur )
        blur_img = img.filter(gaussian_filter)
        return blur_img

if __name__ == "__main__":
    '''
    image = Image.open('stamp.png')
    image = image.convert('RGBA')
    image.putalpha(100)#0~255,255最深，0最淺
    print(image.mode)
    # Transparency
    newImage = []
    for item in image.getdata():
        if item[:3] == (255, 255, 255):
            newImage.append((255, 255, 255, 0))
        else:
            newImage.append(item)
    
    image.putdata(newImage)
    
    
    image.save('./path_to_watermark.png')
    '''
    '''
    firstname = get_str(random.randint(1,4))
    centername = get_str(random.randint(1,4))
    lastname = get_str(random.randint(1,4))
    name = {'first_name': firstname,'center_name': centername,'last_name':lastname}
    gen_stamp_1 = generate_stamp(name)
    fgi_1 = transparent(gen_stamp_1,pha=random.randint(50,240))
    for i in range(10):
        w,h = fgi_1.size
        s = random.randint(2,4)
        new_fgi = fgi_1.resize((s*w,s*h))#隨機放大印章
        background = Image.open("./test.png")
        foreground = random_crop_stamp(new_fgi ,background.size)
        background.paste(foreground, (0, 0), foreground)
        background.show()
    '''
    #fgi_1.save('11.png')
    #fgi_1.show()
    
    
    gen_stamp = Stamp( words_up = get_str(random.randint(7,14)),words_mid=get_str(random.randint(5,7)),words_down=get_num(random.randint(9,14)),img_wl_path="./Generate_Stamp/wl.jpg")
    fgi = gen_stamp.draw_stamp()
    w,h = fgi.size
    #s = random.randint(4,8)
    #new_fgi = fgi.resize((s*w,s*h))#隨機放大印章
    #fgi = random_stamp(fgi)
    #fgi.save('11.png')
    #print(fgi.mode)
    
    #background = Image.open("./test.png")
    #fgi = random_stamp(fgi_1)
    
    for i in range(10):
        s = random.randint(2,8)
        new_fgi = fgi.resize((s*w,s*h))#隨機放大印章
        background = Image.open("./test.png")
        foreground = random_crop_stamp(new_fgi ,background.size)
        background.paste(foreground, (0, 0), foreground)
        background.show()
    
    
    
    #Image.open('./path_to_watermark.png').resize(background.size).convert('RGBA')
    #foreground.putalpha(255)
    

    
    #background = Image.open("./test.png")
    #fgi_1 = random_stamp(fgi_1)
    #foreground = fgi_1.resize(background.size)
    #background.paste(foreground, (0, 0), foreground)
    #background.show()
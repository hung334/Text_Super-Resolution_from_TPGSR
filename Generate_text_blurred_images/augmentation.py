# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:41:57 2022

@author: chen_hung
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageFilter, ImageStat
import random
from Generate_Stamp.generate_complex_stamp import Stamp

class Aug_case1:
    #case1:紅印+壓縮+雜訊+模糊
    def __init__(self, mask=False,interpolation=Image.BICUBIC,aug_probability=0.8):
        
        self.aug_probability = aug_probability
        #self.size = size
        self.interpolation = interpolation
        self.mask = mask

    def __call__(self, img):
        
        if(random.random()<=self.aug_probability):
            x_r = 0.8#random.random()#random.uniform(0.4, 0.59)#random.random()
            if(x_r>0.6):#紅印+壓縮+模糊
                test_img = self.Seal_Enhancement(img)#蓋上紅印,輸出是pil
                #plt.imshow(test_img)
                #plt.show()
                test_img = cv2.cvtColor(np.asarray(test_img), cv2.COLOR_RGB2BGR)#pil to cv
                test_img = self.compress_img(test_img.copy())#壓縮圖像
            else:
                if(x_r>0.3):#紅印+壓縮+雜訊+模糊
                    test_img = self.Seal_Enhancement(img)#蓋上紅印,輸出是pil
                    test_img = cv2.cvtColor(np.asarray(test_img), cv2.COLOR_RGB2BGR)#pil to cv
                else:#壓縮+雜訊+模糊
                    test_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)#pil to cv  
                test_img = self.compress_img(test_img.copy())#壓縮圖像
                g_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)#轉灰階
                test2_img = self.salt_noise(g_img.copy(),SNR=random.uniform(0, 0.8))#白鹽雜訊
                test3_img = self.collage_img(g_img,test2_img,lamb=random.uniform(0.1, 0.5),epoch=random.randint(1, 3))#拼貼圖像
                test_img = cv2.cvtColor(test3_img, cv2.COLOR_GRAY2BGR)#灰階轉RGB
            f_blur_img = self.blur_img(Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)),blur=random.randint(2, 4))
        else:
            #單純只做模糊
            #Image.fromarray(cv2.cvtColor(test3_img, cv2.COLOR_BGR2RGB))
            f_blur_img = self.blur_img(img,blur=random.randint(4, 5))
        #img = f_blur_img.resize(self.size, self.interpolation)
        img = f_blur_img
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        #return img_tensor,f_blur_img
        return f_blur_img
    
    def Seal_Enhancement(self,background):#蓋上印章
        def GBK2312():
            head = random.randint(0xb0, 0xf7)
            body = random.randint(0xa1, 0xf9)   
            val = f'{head:x}{body:x}'
            str = bytes.fromhex(val).decode('gb2312')
            return str
        def get_str(n=4):
            a=''
            for i in range(n):
                a+=GBK2312()
            return a
        def get_num(n=4):
            a=''
            for i in range(n):
                a+=str(random.randint(0,9))
            return a
        def random_crop_stamp(stamp,text_img_size):
            # (left, upper, right, lower)
            s_w,s_h = stamp.size
            t_w,t_h = text_img_size
            #try:
            new_x = random.randint(0,s_w-t_w)
            new_y = random.randint(0,s_h-t_h)
            return stamp.crop((new_x,new_y,new_x+t_w,new_y+t_h))
        try:
            #生成印章
            gen_stamp = Stamp( words_up = get_str(random.randint(7,14)),words_mid=get_str(random.randint(5,7)),words_down=get_num(random.randint(9,14)),img_wl_path="./Generate_Stamp/wl.jpg")
            fgi = gen_stamp.draw_stamp()
            w,h = fgi.size
            s = random.randint(2,8)
            new_fgi = fgi.resize((s*w,s*h))#隨機放大印章
            foreground = random_crop_stamp(new_fgi ,background.size)
            background.paste(foreground, (0, 0), foreground)
            return background
        except:
            return background
    
    def collage_img(self,org_img,ref_img,lamb,epoch):
        # lamb 是反比，值愈小，複製過去的圖範圍愈大
        image_updated = org_img.copy()
        for i in range(epoch):
            lamb = (1-lamb) #反比
            size = org_img.shape
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(size, lamb)
            image_updated[bbx1:bbx2, bby1:bby2] = ref_img[bbx1:bbx2, bby1:bby2].copy()
        return image_updated
    
    def blur_img(self,img,blur = random.randint(1, 5)):
        gaussian_filter = ImageFilter.GaussianBlur(radius=blur )
        blur_img = img.filter(gaussian_filter)
        return blur_img
    
    def compress_img(self,test_img):
        scale = random.randint(2, 4)
        resize_inter = {
            'cv2.INTER_NEAREST':cv2.INTER_NEAREST,
            'cv2.INTER_LINEAR':cv2.INTER_LINEAR,
            'cv2.INTER_AREA':cv2.INTER_AREA,
            'cv2.INTER_CUBIC':cv2.INTER_CUBIC,
            'cv2.INTER_LANCZOS4':cv2.INTER_LANCZOS4 }
        org_H,org_W = test_img.shape[0:2]
        inter_num = random.choice(list(resize_inter.values()))
        re_img = cv2.resize(test_img, (math.ceil(org_W/scale),math.ceil(org_H/scale)), interpolation=inter_num)
        re_img = cv2.resize(re_img, (org_W,org_H), interpolation=inter_num)
        return re_img

    def salt_noise(self,img,SNR=0.8):
        #信噪比
        #SNR = 0.1
        #计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * (1-SNR)
        noiseNum=int((SNR)*img.shape[0]*img.shape[1])
        for i in range(noiseNum):
            randX=random.randint(0,img.shape[0]-1)  
            randY=random.randint(0,img.shape[1]-1)  
            if random.randint(0,1)==0:  
                img[randX,randY]=255  
            else:  
                img[randX,randY]=255
        return img

    def rand_bbox(self,size, lamb):
        """ Generate random bounding box 
        Args:
            - size: [width, breadth] of the bounding box
            - lamb: (lambda) cut ratio parameter
        Returns:
            - Bounding box
        """
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(1. - lamb)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    


if __name__ == "__main__":
    

    
    test_img_path = "./test.png"
    img = Image.open(test_img_path)
    aug = Aug_case1(aug_probability=1)#aug_probability為做[壓縮+噪音+模糊]增強的機率；否則只做[模糊]
    a_img = aug(img)
    plt.imshow(a_img)
    #plt.xticks([])
    #plt.yticks([])
    plt.show()
    
    
    '''
    import os
    
    aug = Aug_case1(aug_probability=0.9)
    
    fh = open('./../val_Images.txt', 'r',encoding="utf_8_sig") 
    imgs = []
    for line in fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1]))#圖片，label
            a = r'V:\data\4TB\chen_hung\TPGSR-main\Text_Data\new_aug_val\{}'.format(words[0])
            if not(os.path.isfile(a)):
                print(words[0])
                test_img_path = "./../Text_Data/HR/{}".format(words[0])
                img = Image.open(test_img_path)
                a_img = aug(img)
                a_img.save(a)
    '''
            
            
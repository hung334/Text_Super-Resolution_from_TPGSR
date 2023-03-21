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


class Aug_case1:
    #case1:壓縮+雜訊+模糊
    def __init__(self, mask=False,interpolation=Image.BICUBIC,aug_probability=0.8):
        
        self.aug_probability = aug_probability
        #self.size = size
        self.interpolation = interpolation
        self.mask = mask

    def __call__(self, img):
        
        if(random.random()<=self.aug_probability):
            test_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)#cv to pil
            test_img = self.compress_img(test_img.copy())#壓縮圖像
            #plt_show(test_img)
            g_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)#轉灰階
            #plt_show(g_img)
            test2_img = self.salt_noise(g_img.copy(),SNR=random.uniform(0, 0.8))#白鹽雜訊
            test3_img = self.collage_img(g_img,test2_img,lamb=random.uniform(0.1, 0.5),epoch=random.randint(1, 3))#拼貼圖像
            test3_img = cv2.cvtColor(test3_img, cv2.COLOR_GRAY2BGR)#灰階轉RGB
            f_blur_img = self.blur_img(Image.fromarray(cv2.cvtColor(test3_img, cv2.COLOR_BGR2RGB)),blur=random.randint(1, 4))
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
    
    
    
    test_img_path = "./t1.jpg"
    
    img = Image.open(test_img_path)
    aug = Aug_case1(aug_probability=1)#aug_probability為做[壓縮+噪音+模糊]增強的機率；否則只做[模糊]
    a_img = aug(img)
    plt.imshow(a_img)
    #plt.xticks([])
    #plt.yticks([])
    plt.show()
    
    
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:49:36 2022

@author: chen_hung
"""
from augmentation import Aug_case1
from TextRecognitionDataGenerator import Generator_str_to_img
import matplotlib.pyplot as plt
import random
import os

if __name__ == '__main__':

    
    '''
    input_string = '公司2021Q1实现营收1070.9亿元，同比增长58.8%'#'详见刊登在上海证券交易所网站'
    HR_img = Generator_str_to_img(input_string)
    aug_img = Aug_case1(aug_probability=0.8) #aug_probability為做[壓縮+噪音+模糊]增強的機率；否則只做[模糊]
    LR_img = aug_img(HR_img)
    
    plt.imshow(LR_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    '''
    
    #part2
    
    fonts_path = './Simplified_Chinese'
    
    for i in os.listdir('./Simplified_Chinese'):
        input_string = '上海证券交易所网站a1'
        HR_img = Generator_str_to_img(input_string,stroke_w=1,font_format=os.path.join(fonts_path,i))
        LR_img =  Aug_case1(aug_probability=0.8)(HR_img)
        #LR_img =  Aug_case1().blur_img(HR_img,blur = 5)
        
        plt.imshow(LR_img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
import os
import torch
import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from PIL import Image, ImageFilter, ImageStat
from torchvision import transforms
import random
from Generate_text_blurred_images.TextRecognitionDataGenerator import Diversity_generator_str_to_img
from random import choice,shuffle

class Aug_case1:
    #case1:壓縮+雜訊+模糊
    def __init__(self, size, mask=False,interpolation=Image.BICUBIC,aug_probability=0.8):
        
        self.aug_probability = aug_probability
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
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
            f_blur_img = self.blur_img(Image.fromarray(cv2.cvtColor(test3_img, cv2.COLOR_BGR2RGB)),blur=random.randint(2, 4))
        else:
            #單純只做模糊
            #Image.fromarray(cv2.cvtColor(test3_img, cv2.COLOR_BGR2RGB))
            f_blur_img = self.blur_img(img,blur=random.randint(4, 5))
        #img = f_blur_img.resize(self.size, self.interpolation)
        img = f_blur_img
        img_tensor = self.toTensor(img)
        # if self.mask:
        #     mask = img.convert('L')
        #     thres = np.array(mask).mean()
        #     mask = mask.point(lambda x: 0 if x > thres else 255)
        #     mask = self.toTensor(mask)
        #     img_tensor = torch.cat((img_tensor, mask), 0)

        #return img_tensor
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
    
    def blur_img(self,img,blur = random.randint(1, 5)):#
        gaussian_filter = ImageFilter.GaussianBlur(radius=blur )
        blur_img = img.filter(gaussian_filter)
        return blur_img
    
    def compress_img(self,test_img):
        scale = random.randint(1, 3)
        resize_inter = {
            'cv2.INTER_NEAREST':cv2.INTER_NEAREST,
            'cv2.INTER_LINEAR':cv2.INTER_LINEAR,
            'cv2.INTER_AREA':cv2.INTER_AREA,
            'cv2.INTER_CUBIC':cv2.INTER_CUBIC,
            'cv2.INTER_LANCZOS4':cv2.INTER_LANCZOS4 }
        org_H,org_W = test_img.shape[0:2]
        re_img = cv2.resize(test_img, (math.ceil(org_W/scale),math.ceil(org_H/scale)), interpolation=random.choice(list(resize_inter.values())))
        
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

def plt_show(image):
    
    if(len(image.shape)==3):
        image = image[:,:,::-1]
    elif(len(image.shape)==2):
        image = image
    
    plt.imshow(image,cmap ='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()

def make_text_data_txt(root,text_label,val_threshold=0.8):#製作myselfdata的切分訓練集和驗證集
    t_fi= open('train_Images.txt','w',encoding="utf_8_sig")
    v_fi= open('val_Images.txt','w',encoding="utf_8_sig")
    
    total=[]
    for img_file in os.listdir(root):
        index = int(img_file.split('.')[0])
        label = text_label[index]
        if(index>30000):#測試集有30000張
            total.append([img_file,label])
    random.shuffle(total)#打亂
    
    total = total[0:1000]
    
    for i,data in enumerate(total):
        if(i<len(total)*val_threshold):
            t_fi.writelines("{} {}\n".format(data[0],data[1]))
        else:
            v_fi.writelines("{} {}\n".format(data[0],data[1]))

    t_fi.close()
    v_fi.close()

class myself_text_datasets_demo(Dataset): #當初做測試的使用，基本上不太會用到
    def __init__(self,root_img,transform=None): 
        #super(MyDataset,self).__init__()
        self.root_img = root_img
        self.transform = transform
        
        #fh = open(datatxt, 'r',encoding="utf_8_sig") 
        imgs = []
        for img_file in os.listdir(self.root_img):
            imgs.append(img_file)#圖片

        self.imgs = imgs

    def __getitem__(self, index):
        
            img_f = self.imgs[index] 
            test_img = Image.open(os.path.join(self.root_img,img_f))
            
            if self.transform is not None:
                test_img = self.transform(test_img) 
                
            return test_img

    def __len__(self): 
        return len(self.imgs)

class myself_text_datasets_for_aug_collate_textzoom_pdfimage(Dataset): #主要用訓練用的，textzoom + pdf_image
    def __init__(self,root_textzoom_HR,root_textzoom_LR,root_pdfimg, textzoom_txt,pdfimg_txt ,aug_trans=None): 
        #super(MyDataset,self).__init__()
        self.root_textzoom_HR = root_textzoom_HR
        self.root_textzoom_LR = root_textzoom_LR
        self.root_pdfimg = root_pdfimg
        self.aug_trans = aug_trans
        
        imgs = []
        pdf_fh = open(pdfimg_txt, 'r',encoding="utf_8_sig") 
        for line in pdf_fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],'Yes'))#圖片
            
        textzoom_fh = open(textzoom_txt, 'r',encoding="utf_8_sig") 
        for line in textzoom_fh :
            line = line.rstrip()
            #words = line.split()
            #print(words)
            imgs.append((line,'No'))#圖片，label
        
        self.imgs = imgs

    def __getitem__(self, index):
        
            fn, label = self.imgs[index]
            #print(fn)
            if(label=='Yes'):
                HR_img = Image.open(os.path.join(self.root_pdfimg,fn))
                LR_img = self.aug_trans(HR_img) 
            else:
                HR_img = Image.open(os.path.join(self.root_textzoom_HR,fn.replace('lr','hr')))
                LR_img = Image.open(os.path.join(self.root_textzoom_LR,fn))

            return HR_img,LR_img,label #輸出是圖片形式
        
    def __len__(self): 
        return len(self.imgs)

class myself_text_datasets_textzoom_easy(Dataset): #主要用於驗證集，myself_val + textzoom_easy
    def __init__(self,root_HR, root_LR ,datatxt, root_textzoom_HR,root_textzoom_LR,textzoom_txt,transform=None): 
        #super(MyDataset,self).__init__()
        self.root_HR = root_HR
        self.root_LR = root_LR
        self.transform = transform
        
        self.root_textzoom_HR = root_textzoom_HR
        self.root_textzoom_LR = root_textzoom_LR
        self.textzoom_txt = textzoom_txt
        
        fh = open(datatxt, 'r',encoding="utf_8_sig") 
        imgs = []
        for line in fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1]))#圖片，label
        th = open(textzoom_txt, 'r',encoding="utf_8_sig") 
        for line in th :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1]))#圖片，label
        
        self.imgs = imgs

    def __getitem__(self, index):
        
            fn, label = self.imgs[index] 
            
            if('image' in fn):
                HR_img = Image.open(os.path.join(self.root_textzoom_HR,fn.replace('lr','hr')))
                LR_img = Image.open(os.path.join(self.root_textzoom_LR,fn))
            else:
                HR_img = Image.open(os.path.join(self.root_HR,fn))
                LR_img = Image.open(os.path.join(self.root_LR,fn))
            
            if self.transform is not None:
                HR_img = self.transform(HR_img) 
                LR_img = self.transform(LR_img)
                
            return HR_img,LR_img,label #輸出是tensor型態

    def __len__(self): 
        return len(self.imgs)

class Diversity_myself_text_datasets_for_aug_collate(Dataset): #主要用於訓練時，多樣性生成myselfdata，就由讀取文本txt
    def __init__(self,datatxt,fonts_path,aug_trans=None): 
        #super(MyDataset,self).__init__()
        #self.context_txt = context_txt
        
        self.aug_trans = aug_trans
        self.fonts_path = fonts_path
        self.fonts_datas = os.listdir(fonts_path)
        
        fh = open(datatxt, 'r',encoding="utf_8_sig") 
        imgs = []
        for line in fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[1]))#label
        self.imgs = imgs

    def __getitem__(self, index):
        
            label = self.imgs[index] 
            if(random.random()>0.4):
                reg = choice(self.imgs)
                new = label+reg
                label = self.shuffle_str(new)
                try:
                    label = label[0:random.randint(5,len(label)-1)]
                except:
                    label = label[0:random.randint(1,len(label)-1)]
    
            HR_img = Diversity_generator_str_to_img(label,self.fonts_path,self.fonts_datas)
            LR_img = self.aug_trans(HR_img) 
                
            return HR_img,LR_img,label #輸出是圖片形式
        
    def __len__(self): 
        return len(self.imgs)
    def shuffle_str(self,s):
        # 将字符串转换成列表
        str_list = list(s)
        # 调用random模块的shuffle函数打乱列表
        shuffle(str_list)
        # 将列表转字符串
        return ''.join(str_list)

class myself_text_datasets_for_aug_collate(Dataset): #導入多尺度，應用collate_fn 來解決包裝
    def __init__(self,root_HR, datatxt, aug_trans=None): 
        #super(MyDataset,self).__init__()
        self.root_HR = root_HR
        self.aug_trans = aug_trans
        
        fh = open(datatxt, 'r',encoding="utf_8_sig") 
        imgs = []
        for line in fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1]))#圖片，label
        self.imgs = imgs

    def __getitem__(self, index):
        
            fn, label = self.imgs[index] 
            HR_img = Image.open(os.path.join(self.root_HR,fn))
            LR_img = self.aug_trans(HR_img) 
            #HR_img = self.RN(HR_img)

            return HR_img,LR_img,label #輸出是圖片形式
        
    def __len__(self): 
        return len(self.imgs)

class myself_text_datasets_for_aug(Dataset): #當初測試添加數據增強，基本上不太用到已有更新的
    def __init__(self,root_HR, root_LR ,datatxt, transform=None,RN=None): 
        #super(MyDataset,self).__init__()
        self.root_HR = root_HR
        self.root_LR = root_LR
        self.transform = transform
        self.RN = RN
        
        fh = open(datatxt, 'r',encoding="utf_8_sig") 
        imgs = []
        for line in fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1]))#圖片，label
        self.imgs = imgs

    def __getitem__(self, index):
        
            fn, label = self.imgs[index] 
            HR_img = Image.open(os.path.join(self.root_HR,fn))
            #LR_img = Image.open(os.path.join(self.root_LR,fn))
            LR_img = self.transform(HR_img) 
            HR_img = self.RN(HR_img)
            

                
            return HR_img,LR_img,label #輸出是tensor型態

    def __len__(self): 
        return len(self.imgs)

class myself_text_datasets(Dataset): #最剛開始撰寫讀取myselfdata-LRb5&LRb4
    def __init__(self,root_HR, root_LR ,datatxt, transform=None): 
        #super(MyDataset,self).__init__()
        self.root_HR = root_HR
        self.root_LR = root_LR
        self.transform = transform
        
        fh = open(datatxt, 'r',encoding="utf_8_sig") 
        imgs = []
        for line in fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1]))#圖片，label
        self.imgs = imgs

    def __getitem__(self, index):
        
            fn, label = self.imgs[index] 
            HR_img = Image.open(os.path.join(self.root_HR,fn))
            LR_img = Image.open(os.path.join(self.root_LR,fn))
            
            if self.transform is not None:
                HR_img = self.transform(HR_img) 
                LR_img = self.transform(LR_img)
                
            return HR_img,LR_img,label #輸出是tensor型態

    def __len__(self): 
        return len(self.imgs)

class Collate_resizeNormalize(object): #用於 Collate_fn   主要做多尺度的轉變，有padding 和 resize
    def __init__(self,imgH=48,imgW=100,keep_ratio=True, min_ratio=1,use_padding=True,mask=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.use_padding = use_padding
        self.mask = mask
        
    def __call__(self, batch):

        HR_imgs,LR_imgs,labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        
        if self.keep_ratio:
            ratios = []
            for image in HR_imgs:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
        
        imgW = imgW-1 if imgW%2!=0 else imgW
        if(self.use_padding):
            transform = padding_resizeNormalize(size=(imgW, imgH),mask=self.mask )
        else:
            transform = resizeNormalize(size=(imgW, imgH),mask=self.mask )
        #resizeNormalize(size=(imgW, imgH),mask=True)
        
        LR_images = [transform(image) for image in LR_imgs]
        LR_images = torch.stack(LR_images, 0)
        #LR_images = torch.cat([t.unsqueeze(0) for t in LR_images], 0)
        
        HR_images = [transform(image) for image in HR_imgs]
        HR_images = torch.stack(HR_images, 0)
        #HR_images = torch.cat([t.unsqueeze(0) for t in HR_images], 0)

        return HR_images,LR_images ,labels


class padding_resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img):
        
        imgW, imgH = self.size
        img = transforms.Resize(imgH, interpolation=Image.BICUBIC)(img)
        img_w,img_h = img.size
        new_img = Image.new('RGB',self.size,(255,255,255))
        new_img.paste(img,(0,0,img_w,img_h))
        
        img_tensor = self.toTensor(new_img)
        if self.mask:
            mask = new_img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor

class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor

if __name__ == "__main__":
    
    
    
    root = './Text_Data/HR'
    #text_label = np.load('./record_string.npy')
    #make_text_data_txt(root,text_label,val_threshold=0.8)
    '''
    trc = transforms.Compose([ 
                               transforms.ToTensor()])
    tensor_to_Pil = transforms.ToPILImage()
    
    train_data = myself_text_datasets(root_HR='./Text_Data/HR', root_LR='./Text_Data/LR_b5' ,datatxt='./myself_data/val_Images.txt',transform=resizeNormalize((128,32),True))
    train_dataloader = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=0)
    
    for step,(HR,LR,label) in enumerate(train_dataloader):
        print(label)
        pass
            
    pass
    '''
    
    
    tensor_to_Pil = transforms.ToPILImage()
    trs_aug = Aug_case1((32,64), mask=False,aug_probability=1) #宣告數據增強法，size的部分不會用到
    train_data = myself_text_datasets_for_aug_collate(root_HR='./Text_Data/HR',datatxt='./myself_data/train_Images.txt',aug_trans=trs_aug) #回傳 增強後的圖像。
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True,num_workers=int(0),collate_fn=Collate_resizeNormalize(imgH=32))
    for step,(HR,LR,label) in enumerate(train_dataloader):
        
        for i in LR :
            plt.imshow(tensor_to_Pil(i))
            plt.show()
        #print(label)


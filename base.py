import torch
import sys
import os
from tqdm import tqdm
import math
import torch.nn as nn
import torch.optim as optim
from IPython import embed
import math
import cv2
import string
from PIL import Image
import PIL
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict
import ptflops

from model import tsrn, bicubic, srcnn, vdsr, srresnet, edsr, esrgan, rdn, lapsrn
from model import recognizer
from model import moran
from model import crnn
from dataset import lmdbDataset, \
    alignCollate_real, ConcatDataset, lmdbDataset_real, \
    alignCollate_syn, lmdbDataset_mix, alignCollateW2V_real, \
    lmdbDatasetWithW2V_real, alignCollatec2f_real, lmdbDataset_realIC15, \
    alignCollate_realWTL, alignCollate_realWTL_withcrop, alignCollate_realWTLAMask, \
lmdbDatasetWithMask_real, lmdbDataset_realIC15TextSR, lmdbDataset_realCOCOText, lmdbDataset_realSVT, \
lmdbDataset_realBadSet, alignCollate_syn_random_reso, lmdbDataset_realIIIT, lmdbDataset_realForTest
from loss import gradient_loss, percptual_loss, image_loss, semantic_loss

from utils.labelmaps import get_vocabulary, labels2strs

sys.path.append('../')
from utils import util, ssim_psnr, utils_moran, utils_crnn
import dataset.dataset as dataset

from rec.RecModel import RecModel

from myself_text_dataloader import resizeNormalize,myself_text_datasets,myself_text_datasets_demo,myself_text_datasets_for_aug,Aug_case1,myself_text_datasets_for_aug_collate,Collate_resizeNormalize
from myself_text_dataloader import myself_text_datasets_for_aug_collate_textzoom_pdfimage,myself_text_datasets_textzoom_easy
from myself_text_dataloader import Diversity_myself_text_datasets_for_aug_collate

# backbone_dict = {"SVTR":SVTRNet,"MobileNetV1Enhance":MobileNetV1Enhance}
# neck_dict = {'PPaddleRNN': SequenceEncoder, 'Im2Seq': Im2Seq,'None':Im2Im}
# head_dict = {'CTC': CTC,'Multi':MultiHead}

# class RecModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         assert 'in_channels' in config, 'in_channels must in model config'
#         backbone_type = config.backbone.pop('type')
#         assert backbone_type in backbone_dict, f'backbone.type must in {backbone_dict}'
#         self.backbone = backbone_dict[backbone_type](config.in_channels, **config.backbone)

#         neck_type = config.neck.pop('type')
#         assert neck_type in neck_dict, f'neck.type must in {neck_dict}'
#         self.neck = neck_dict[neck_type](self.backbone.out_channels, **config.neck)

#         head_type = config.head.pop('type')
#         assert head_type in head_dict, f'head.type must in {head_dict}'
#         self.head = head_dict[head_type](self.neck.out_channels, **config.head)

#         self.name = f'RecModel_{backbone_type}_{neck_type}_{head_type}'

#     def load_3rd_state_dict(self, _3rd_name, _state):
#         self.backbone.load_3rd_state_dict(_3rd_name, _state)
#         self.neck.load_3rd_state_dict(_3rd_name, _state)
#         self.head.load_3rd_state_dict(_3rd_name, _state)

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.neck(x)
#         x = self.head(x)
#         return x

def load_trained_model(create_model, model_path):
    
    state_dict = torch.load(model_path)        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k]=v

    create_model.load_state_dict(new_state_dict)

    return create_model

class TextBase(object):
    def __init__(self, config, args, opt_TPG=None):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        self.opt_TPG = opt_TPG

        # print("data goes here:")

        # lmdbDataset_realIC15

        picked = "SVT"

        syn_datasets = {
            "IC15": lmdbDataset_realIC15,
            "SynthText": lmdbDataset_realIC15,
            "SVT": lmdbDataset_realSVT,
            "COCO-Text": lmdbDataset_realCOCOText,
            "BadSet": lmdbDataset_realBadSet,
            "IIIT5K": lmdbDataset_realIIIT
        }

        if self.args.random_reso:
            align_type = "random"
        else:
            align_type = "fixed"

        align_types = {
            "random": alignCollate_syn_random_reso,
            "fixed": alignCollate_syn
        }

        if self.args.syn:
            if self.args.arch == 'tsrn_tl_cascade':
                # self.align_collate = alignCollate_syn
                # self.load_dataset = lmdbDataset

                # self.align_collate_val = alignCollate_real
                # self.load_dataset_val = lmdbDataset_real

                self.align_collate = align_types[align_type] #alignCollate_real # align_types[align_type]#alignCollate_syn
                self.load_dataset = syn_datasets[picked]

                # self.align_collate_val = alignCollate_syn
                # self.load_dataset_val = lmdbDataset_realCOCOText

                self.align_collate_val = align_types[align_type] #alignCollate_realWTL # align_types[align_type]#alignCollate_syn
                self.load_dataset_val = syn_datasets[picked] # lmdbDataset_real #

            elif self.args.arch == 'srcnn_tl':
                self.align_collate = align_types[align_type]
                self.load_dataset = syn_datasets[picked]

                self.align_collate_val = align_types[align_type]
                self.load_dataset_val = syn_datasets[picked]

            elif self.args.arch == 'srresnet_tl':
                self.align_collate = align_types[align_type]
                self.load_dataset = syn_datasets[picked]

                self.align_collate_val = align_types[align_type]
                self.load_dataset_val = syn_datasets[picked]
            else:
                self.align_collate = align_types[align_type]
                self.load_dataset = syn_datasets[picked]

                self.align_collate_val = align_types[align_type]
                self.load_dataset_val = syn_datasets[picked]
            

        elif self.args.mixed:
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_mix

        elif self.args.ic15sr:
            if self.args.arch == 'tsrn_tl_cascade':
                # self.align_collate = alignCollate_syn
                # self.load_dataset = lmdbDataset

                # self.align_collate_val = alignCollate_real
                # self.load_dataset_val = lmdbDataset_real

                self.align_collate = alignCollate_realWTLAMask
                self.load_dataset = lmdbDataset_realIC15TextSR

                self.align_collate_val = alignCollate_realWTL
                self.load_dataset_val = lmdbDataset_realIC15TextSR
            else:
                self.align_collate = alignCollate_real
                self.load_dataset = lmdbDataset_realIC15TextSR

                self.align_collate_val = alignCollate_real
                self.load_dataset_val = lmdbDataset_realIC15TextSR


        else:
            # print("data goes here:")
            if self.args.arch == "tsrn":
                self.align_collate = alignCollate_realWTLAMask # alignCollate_real
                self.load_dataset = lmdbDataset_real

                self.align_collate_val = self.align_collate
                self.load_dataset_val = self.load_dataset

                # if self.args.go_test:
                # self.load_dataset_val = lmdbDataset_realIC15
                # self.load_dataset_val = lmdbDataset_real
                self.load_dataset_val = lmdbDataset_realForTest#lmdbDataset_real
            elif self.args.arch == "sem_tsrn":
                self.align_collate = alignCollateW2V_real
                self.load_dataset = lmdbDatasetWithW2V_real

                self.load_dataset_val = lmdbDatasetWithW2V_real

            elif self.args.arch == "tsrn_c2f":
                self.align_collate = alignCollatec2f_real
                self.load_dataset = lmdbDataset_real

            elif self.args.arch == "tsrn_tl":
                self.align_collate = alignCollate_realWTL
                self.load_dataset = lmdbDataset_real

                self.align_collate_val = alignCollate_realWTL
                self.load_dataset_val = lmdbDataset_real

            elif self.args.arch == 'tsrn_tl_wmask':
                self.align_collate = alignCollate_realWTLAMask
                self.load_dataset = lmdbDataset_real

                self.align_collate_val = alignCollate_realWTL
                self.load_dataset_val = lmdbDataset_real
            elif self.args.arch == 'tsrn_tl_cascade':
                self.align_collate = alignCollate_realWTLAMask
                self.load_dataset = lmdbDataset_real

                self.align_collate_val = alignCollate_realWTL
                self.load_dataset_val = lmdbDataset_real#lmdbDataset_realForTest #lmdbDataset_real

            elif self.args.arch == 'srcnn_tl':
                self.align_collate = alignCollate_realWTLAMask
                self.load_dataset = lmdbDataset_real

                self.align_collate_val = alignCollate_realWTL
                self.load_dataset_val = lmdbDataset_real

            elif self.args.arch == 'srresnet_tl':
                self.align_collate = alignCollate_realWTLAMask
                self.load_dataset = lmdbDataset_real

                self.align_collate_val = alignCollate_realWTL
                self.load_dataset_val = lmdbDataset_real

            elif self.args.arch == 'rdn_tl':
                self.align_collate = alignCollate_realWTLAMask
                self.load_dataset = lmdbDataset_real

                self.align_collate_val = alignCollate_realWTL
                self.load_dataset_val = lmdbDataset_real

            elif self.args.arch == 'vdsr_tl':
                self.align_collate = alignCollate_realWTLAMask
                self.load_dataset = lmdbDataset_real

                self.align_collate_val = alignCollate_realWTL
                self.load_dataset_val = lmdbDataset_real

            else:

                # print("go here:")

                self.align_collate = alignCollate_real
                self.load_dataset = lmdbDataset_real

                self.align_collate_val = self.align_collate
                self.load_dataset_val = self.load_dataset

        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.val_batch_size = args.val_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len
        self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)
    
    def get_train_Diversity_myself_data(self):#訓練集:多樣性生成(字型、字體寬度、背景......)
        
        print('*'*30,'get_train_Diversity_myself_data')
        
        fonts_path = './Generate_text_blurred_images/Simplified_Chinese'
        fonts_datas = os.listdir(fonts_path)
        
        cfg = self.config.TRAIN
        trs_aug = Aug_case1((cfg.width,cfg.height), mask=False,aug_probability=self.args.aug_prob) #宣告數據增強法，size的部分不會用到
        train_data = Diversity_myself_text_datasets_for_aug_collate(datatxt='./myself_data/train_Images.txt',fonts_path=fonts_path,aug_trans=trs_aug) #回傳 增強後的圖像。
        train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=int(cfg.workers),collate_fn=Collate_resizeNormalize(imgH=cfg.height,mask=self.mask))
        
        return train_data, train_dataloader

    def get_train_textzoom_pdfimg(self):#訓練集:textzoom+清晰pdf
        
        textzoom_HR = r'./text_zoom_img/HR'
        textzoom_LR = r'./text_zoom_img/LR'
        root_pdfimg = r'./pdf_image_datasets'
        textzoom_txt = './datasets_txt/textzoom_LR.txt'
        pdfimg_txt = './datasets_txt/pdf_image_datasets.txt'
            
        cfg = self.config.TRAIN
        trs_aug = Aug_case1((cfg.width,cfg.height), mask=False,aug_probability=self.args.aug_prob) #宣告數據增強法，size的部分不會用到
        train_data = myself_text_datasets_for_aug_collate_textzoom_pdfimage(root_textzoom_HR=textzoom_HR,root_textzoom_LR=textzoom_LR,root_pdfimg=root_pdfimg, textzoom_txt=textzoom_txt,pdfimg_txt=pdfimg_txt,aug_trans=trs_aug) #回傳 增強後的圖像。
        train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=int(cfg.workers),collate_fn=Collate_resizeNormalize(imgH=cfg.height,mask=self.mask,use_padding=self.args.use_padding))
        
        return train_data, train_dataloader

    def get_val_myself_data_textzoom_easy(self):#驗證集:textzoom_easy+myselfdata
        cfg = self.config.TRAIN
        
        textzoom_HR = r'./text_zoom_img/test/easy/HR'
        textzoom_LR = r'./text_zoom_img/test/easy/LR'
        textzoom_txt = './text_zoom_img/test/easy/label.txt'
        
        dataset_list = []
        loader_list = []
        if True:
        #for data_dir_ in cfg.VAL.val_data_dir:
            val_data = myself_text_datasets_textzoom_easy(root_HR='./Text_Data/HR', root_LR='./Text_Data/LR_b5' ,datatxt='./myself_data/val_Images.txt',root_textzoom_HR=textzoom_HR,root_textzoom_LR=textzoom_LR,textzoom_txt=textzoom_txt,transform=resizeNormalize((cfg.width,cfg.height),self.mask))
            val_dataloader = torch.utils.data.DataLoader(val_data,batch_size=self.val_batch_size,shuffle=False,num_workers=int(cfg.workers))
            dataset_list.append(val_data)
            loader_list.append(val_dataloader)
        
        return dataset_list, loader_list

    def get_train_myself_dat_aug_case1_multisize(self):#標準版
        
        print('*'*30,'get_train_myself_dat_aug_case1_multisize')
        
        cfg = self.config.TRAIN
        trs_aug = Aug_case1((cfg.width,cfg.height), mask=False,aug_probability=self.args.aug_prob) #宣告數據增強法，size的部分不會用到
        train_data = myself_text_datasets_for_aug_collate(root_HR='./Text_Data/HR',datatxt='./myself_data/train_Images.txt',aug_trans=trs_aug) #回傳 增強後的圖像。
        train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=int(cfg.workers),collate_fn=Collate_resizeNormalize(imgH=cfg.height,mask=self.mask))
        
        return train_data, train_dataloader
    
    def get_train_myself_dat_aug_case1(self):
        cfg = self.config.TRAIN
        trs_aug = Aug_case1((cfg.width,cfg.height), mask=self.mask,aug_probability=self.args.aug_prob)
        train_data = myself_text_datasets_for_aug(root_HR='./Text_Data/HR', root_LR='./Text_Data/LR_b5' ,datatxt='./myself_data/train_Images.txt',transform=trs_aug,RN=resizeNormalize((cfg.width,cfg.height),self.mask))
        train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=int(cfg.workers))
        
        return train_data, train_dataloader
    def get_train_myself_data(self):
        cfg = self.config.TRAIN
        train_data = myself_text_datasets(root_HR='./Text_Data/HR', root_LR='./Text_Data/LR_b5' ,datatxt='./myself_data/train_Images.txt',transform=resizeNormalize((cfg.width,cfg.height),self.mask))
        train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=int(cfg.workers))
        
        return train_data, train_dataloader
    
    def get_val_myself_data(self):
        
        print("*"*30,'get_val_myself_data')
        
        cfg = self.config.TRAIN
        
        dataset_list = []
        loader_list = []
        if True:
        #for data_dir_ in cfg.VAL.val_data_dir:
            val_data = myself_text_datasets(root_HR='./Text_Data/HR', root_LR='./Text_Data/LR_b5' ,datatxt='./myself_data/val_Images.txt',transform=resizeNormalize((cfg.width,cfg.height),self.mask))
            val_dataloader = torch.utils.data.DataLoader(val_data,batch_size=self.val_batch_size,shuffle=False,num_workers=int(cfg.workers))
            dataset_list.append(val_data)
            loader_list.append(val_dataloader)
        
        return dataset_list, loader_list
    
    def get_test_myself_data(self):
        cfg = self.config.TRAIN
        test_data = myself_text_datasets_demo(root_img=self.args.demo_dir,transform=resizeNormalize((cfg.width,cfg.height),self.mask))
        test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=self.batch_size,shuffle=False,num_workers=int(cfg.workers))
        
        return test_dataloader
    
    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    self.load_dataset(root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask, train=True),
            drop_last=True)
        return train_dataset, train_loader

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        self.args.test_data_dir

        if self.args.go_test:
            test_dataset = self.load_dataset_val(root=dir_,
                                             voc_type=cfg.voc_type,
                                             max_len=cfg.max_len,
                                             test=True,
                                             )
        else:
            test_dataset = self.load_dataset_val(root=dir_,  #load_dataset
                                             voc_type=cfg.voc_type,
                                             max_len=cfg.max_len,
                                             test=True,
                                             )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=int(cfg.workers),
            collate_fn=self.align_collate_val(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask, train=False),
            drop_last=False)
        return test_dataset, test_loader
    
    def generator_init_myself(self):
        #torch.cuda.set_device(0)
        cfg = self.config.TRAIN
        model = tsrn.TSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                 STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                 hidden_units=self.args.hd_u)
        image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
        
        # with torch.cuda.device(0):
        #     # input = torch.randn(1, 3, 16, 64).to(self.device)
        #     # tp_in = torch.randn(1, 37, 1, 25).to(self.device)
        #     # net = models.densenet161()
        #     macs, params = ptflops.get_model_complexity_info(model, (4, 16, 64), as_strings=True,
        #                                                      print_per_layer_stat=True, verbose=True)
        #     print("---------------- SR Module -----------------")
        #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        #     print("--------------------------------------------")
            
        print('loading pre-trained model from %s ' % self.resume)
        #model.load_state_dict(torch.load(self.resume)['state_dict_G'])
        if(self.resume != ""):
            model.load_state_dict({k: v for k, v in torch.load(self.resume)['state_dict_G'].items()})
            print("model load ok~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        #model = torch.nn.DataParallel(model, device_ids=[0])
        #model.load_state_dict({'module.' + k: v for k, v in torch.load(self.resume)['state_dict_G'].items()})
        
        # state_dict = torch.load(self.resume)        
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     if 'module' in k:
        #         k = k.replace('module.', '')
        #     new_state_dict[k]=v
        # model.load_state_dict(new_state_dict)
        #model = model.to('cuda:0')
        #model = model.module.to('cpu')
        
        return {'model': model, 'crit': image_crit}
        
    
    def generator_init(self, iter=-1):
        cfg = self.config.TRAIN
        if self.args.arch == 'tsrn':
            model = tsrn.TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                       STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
        elif self.args.arch == 'tsrn_c2f':
            model = tsrn.TSRN_C2F(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                       STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])

        elif self.args.arch == 'sem_tsrn':
            model = tsrn.SEM_TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                              STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_loss_com = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
            semantic_loss_com = semantic_loss.SemanticLoss()
            image_crit = {"image_loss": image_loss_com, "semantic_loss": semantic_loss_com}
        elif self.args.arch == 'tsrn_tl':
            model = tsrn.TSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                  STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                  hidden_units=self.args.hd_u)

            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])

        elif self.args.arch == 'tsrn_tl_wmask':
            model = tsrn.TSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                  STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                  hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])

        elif self.args.arch == 'tsrn_tl_cascade':
            model = tsrn.TSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                 STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                 hidden_units=self.args.hd_u)

            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
        elif self.args.arch == 'bicubic' and self.args.test:
            model = bicubic.BICUBIC(scale_factor=self.scale_factor)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srcnn':
            model = srcnn.SRCNN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'vdsr':
            model = vdsr.VDSR(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srres':
            model = srresnet.SRResNet(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                      STN=self.args.STN, mask=self.mask)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'esrgan':
            model = esrgan.RRDBNet(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'rdn':
            model = rdn.RDN(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'edsr':
            model = edsr.EDSR(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'lapsrn':
            model = lapsrn.LapSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = lapsrn.L1_Charbonnier_loss()

        elif self.args.arch == 'srcnn_tl':
            model = srcnn.SRCNN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()

        elif self.args.arch == 'srresnet_tl':
            model = srresnet.SRResNet_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                      STN=self.args.STN, mask=self.mask)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'rdn_tl':
            model = rdn.RDN_TL(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'vdsr_tl':
            model = vdsr.VDSR_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        else:
            raise ValueError

        with torch.cuda.device(0):
            # input = torch.randn(1, 3, 16, 64).to(self.device)
            # tp_in = torch.randn(1, 37, 1, 25).to(self.device)
            # net = models.densenet161()
            macs, params = ptflops.get_model_complexity_info(model, (4, 16, 64), as_strings=True,
                                                             print_per_layer_stat=False, verbose=True) #print_per_layer_stat=True
            #print("---------------- SR Module -----------------")
            #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            #print("--------------------------------------------")

        if self.args.arch != 'bicubic':
            model = model.to(self.device)
            if self.args.arch == 'sem_tsrn':
                for k in image_crit.keys():
                    image_crit[k] = image_crit[k].to(self.device)
            else:
                image_crit.to(self.device)
            if cfg.ngpu > 1:

                print("multi_gpu", self.device)

                model = torch.nn.DataParallel(model, device_ids=range(cfg.ngpu))

                if self.args.arch == 'sem_tsrn':
                    for k in image_crit.keys():
                        image_crit[k] = torch.nn.DataParallel(image_crit[k], device_ids=range(cfg.ngpu))
                else:
                    image_crit = torch.nn.DataParallel(image_crit, device_ids=range(cfg.ngpu))

            if self.resume is not '':
                print('loading pre-trained model from %s ' % self.resume)
                if self.config.TRAIN.ngpu == 1:
                    # if is dir, we need to initialize the model list
                    if os.path.isdir(self.resume):
                        model.load_state_dict(
                            torch.load(
                                os.path.join(self.resume, "model_best_" + str(iter) + ".pth")
                            )['state_dict_G']
                            )
                    else:
                        model.load_state_dict(torch.load(self.resume)['state_dict_G'])
                else:

                    if os.path.isdir(self.resume):
                        model.load_state_dict(
                            {'module.' + k: v for k, v in torch.load(
                                os.path.join(self.resume, "model_best_" + str(iter) + ".pth")
                            )['state_dict_G'].items()}
                            )
                    else:
                        model.load_state_dict(
                        {'module.' + k: v for k, v in torch.load(self.resume)['state_dict_G'].items()})
        return {'model': model, 'crit': image_crit}

    def optimizer_init(self, model, recognizer=None):
        cfg = self.config.TRAIN

        # print("recognizer:", recognizer)

        if not recognizer is None:

            if type(recognizer) == list:
                if cfg.optimizer == "Adam":

                    # print("model:", type(model.parameters()), model.parameters())
                    rec_params = []
                    model_params = []
                    for recg in recognizer:
                        rec_params += list(recg.parameters())

                    if type(model) == list:
                        for m in model:
                            model_params += list(m.parameters())
                    else:
                        model_params = list(model.parameters())

                    optimizer = optim.Adam(model_params + rec_params, lr=cfg.lr,
                                           betas=(cfg.beta1, 0.999))
                elif cfg.optimizer == "SGD":
                    optimizer = optim.SGD(list(model.parameters()) + rec_params, lr=cfg.lr,
                                          momentum=0.9)
            else:
                if cfg.optimizer == "Adam":

                    # print("model:", type(model.parameters()), model.parameters())
                    model_params = []
                    if type(model) == list:
                        for m in model:
                            model_params += list(m.parameters())
                    else:
                        model_params = list(model.parameters())

                    optimizer = optim.Adam(model_params + list(recognizer.parameters()), lr=cfg.lr,
                                           betas=(cfg.beta1, 0.999))
                elif cfg.optimizer == "SGD":
                    optimizer = optim.SGD(list(model.parameters()) + list(recognizer.parameters()), lr=cfg.lr,
                                          momentum=0.9)

        else:
            model_params = []
            if type(model) == list:
                for m in model:
                    model_params += list(m.parameters())
            else:
                model_params = list(model.parameters())

            if cfg.optimizer == "Adam":
                optimizer = optim.Adam(model_params, lr=cfg.lr,
                                       betas=(cfg.beta1, 0.999))
            elif cfg.optimizer == "SGD":
                optimizer = optim.SGD(model_params, lr=cfg.lr,
                                      momentum=0.9)

        return optimizer



    def tripple_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, index):
        for i in (range(self.config.TRAIN.VAL.n_vis)):
            # embed()
            tensor_in = image_in[i][:3,:,:]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3,:,:]
            tensor_target = image_target[i][:3,:,:]
            images = ([tensor_in, tensor_out.cpu(), tensor_target.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            out_root = os.path.join('./demo', self.vis_dir)
            if not os.path.exists(out_root):
                os.mkdir(out_root)
            out_path = os.path.join(out_root, str(index))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
            im_name = im_name.replace('/', '')
            if index is not 0:
                torchvision.utils.save_image(vis_im, os.path.join(out_path, im_name), padding=0)
    def training_test_display_myself(self, image_in, image_out, pred_str_lr, pred_str_sr, label_strs, index,epoch, iters):
        for i in (range(min(image_in.shape[0], self.config.TRAIN.VAL.n_vis) )):
            # embed()
            tensor_in = image_in[i][:3,:,:]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_out.shape[-2], image_out.shape[-1])),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3,:,:].cpu()
            show_out = transforms.ToPILImage()(tensor_out)
            #print(tensor_in.shape)
            show_out = show_out.resize((tensor_in.shape[-1],tensor_in.shape[-2]),PIL.Image.BICUBIC)
            images = ([tensor_in, tensor_out.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            #out_root = os.path.join('./display', self.vis_dir)
            out_root = os.path.join('ckpt', self.vis_dir) # self.vis_dir#'./display'
            #show_out.save(os.path.join(out_root, "{}_{}_{}".format(label_strs,epoch, iters)))
            #if not os.path.exists(out_root):
            #    os.mkdir(out_root)
            #out_path = os.path.join(out_root, str(index))
            #if not os.path.exists(out_path):
            #    os.mkdir(out_path)
            im_name = "vis_{}_{}_{}".format(epoch, iters,label_strs)#pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
            im_name = im_name.replace('/', '')
            #print("im_name",im_name)
            if index is not 0:
                torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)
                
    def test_display_myself(self, image_in, image_out, pred_str_lr, pred_str_sr, label_strs, index):
        for i in (range(min(image_in.shape[0], self.config.TRAIN.VAL.n_vis) )):
            # embed()
            tensor_in = image_in[i][:3,:,:]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_out.shape[-2], image_out.shape[-1])),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3,:,:]
            show_out = transforms.ToPILImage()(tensor_out)
            #print(tensor_in.shape)
            show_out = show_out.resize((tensor_in.shape[-1],tensor_in.shape[-2]),PIL.Image.BICUBIC)
            
            '''
            print('tensor_in', type(tensor_in), tensor_in.dtype)
            print(torch.min(tensor_in), torch.max(tensor_in))
            print('tensor_out', type(tensor_out), tensor_out.dtype)
            print(torch.min(tensor_out), torch.max(tensor_out))
            print('tensor_in_mean', torch.mean(tensor_in))
            print('tensor_out_mean', torch.mean(tensor_out))
            '''
            images = ([tensor_in, tensor_out.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            #out_root = os.path.join('./display', self.vis_dir)
            out_root = self.vis_dir#'./display'
            show_out.save(os.path.join(out_root, "{}".format(label_strs)))
            #if not os.path.exists(out_root):
            #    os.mkdir(out_root)
            #out_path = os.path.join(out_root, str(index))
            #if not os.path.exists(out_path):
            #    os.mkdir(out_path)
            im_name = "vis_{}".format(label_strs)#pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
            im_name = im_name.replace('/', '')
            if index is not 0:
                torchvision.utils.save_image(vis_im, os.path.join(self.vis_dir, im_name), padding=0)
                #torchvision.utils.save_image(vis_im, os.path.join(out_root, '{}_'.format(label_strs)), padding=0)
            
    def test_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, str_filt):
        visualized = 0
        for i in (range(image_in.shape[0])):
            if True:
                if (str_filt(pred_str_lr[i], 'lower') != str_filt(label_strs[i], 'lower')) and \
                        (str_filt(pred_str_sr[i], 'lower') == str_filt(label_strs[i], 'lower')):
                    visualized += 1
                    tensor_in = image_in[i].cpu()
                    tensor_out = image_out[i].cpu()
                    tensor_target = image_target[i].cpu()
                    transform = transforms.Compose(
                        [transforms.ToPILImage(),
                         transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                         transforms.ToTensor()]
                    )
                    tensor_in = transform(tensor_in)
                    images = ([tensor_in, tensor_out, tensor_target])
                    vis_im = torch.stack(images)
                    vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
                    out_root = os.path.join('./demo', self.vis_dir)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
                    im_name = im_name.replace('/', '')
                    torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)
        return visualized

    def save_checkpoint(self, netG_list, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, recognizer=None):
        ckpt_path = os.path.join('ckpt', self.vis_dir)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        print("Into saving checkpoints...")

        for i in range(len(netG_list)):
            netG = netG_list[i]
            save_dict = {
                'state_dict_G': netG.module.state_dict(),
                'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                         'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
                'best_history_res': best_acc_dict,
                'best_model_info': best_model_info,
                'param_num': sum([param.nelement() for param in netG.module.parameters()]),
                'converge': converge_list,
            }

            if is_best:
                #torch.save(save_dict, os.path.join(ckpt_path, 'model_best_' + str(i) + '.pth'))
                torch.save(save_dict, os.path.join(ckpt_path, 'model_best_{}_{}_{}_{}.pth'.format(str(i),epoch, iters,math.ceil(save_dict['best_history_res']['myself_data']*100))))
                
            else:
                torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))
        
        #不需要儲存文本辨識的模型
        '''
        if is_best:
            # torch.save(save_dict, os.path.join(ckpt_path, 'model_best.pth'))
            if not recognizer is None:
                if type(recognizer) == list:
                    for i in range(len(recognizer)):
                        torch.save(recognizer[i].state_dict(), os.path.join(ckpt_path, 'recognizer_best_' + str(i) + '.pth'))
                else:
                    torch.save(recognizer.state_dict(), os.path.join(ckpt_path, 'recognizer_best.pth'))
        else:
            # torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))
            if not recognizer is None:
                if type(recognizer) == list:
                    for i in range(len(recognizer)):
                        torch.save(recognizer[i].state_dict(), os.path.join(ckpt_path, 'recognizer_' + str(i) + '.pth'))
                else:
                    torch.save(recognizer.state_dict(), os.path.join(ckpt_path, 'recognizer.pth'))
        '''

    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)

        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100

        if self.args.random_reso:
            batch_size = len(imgs_input)
            new_input = []
            for img in imgs_input:
                new_input.append(torch.nn.functional.interpolate(img[:, :3, ...], (32, in_width), mode='bicubic'))
            imgs_input = torch.cat(new_input, 0)
        else:
            batch_size = imgs_input.shape[0]
            imgs_input = torch.nn.functional.interpolate(imgs_input[:, :3, ...], (32, in_width), mode='bicubic')

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def Paddle_OCR_init(self,use_cuda=True):
        #from rec.RecModel import RecModel
        from addict import Dict as AttrDict
        
        cfg = self.config.TRAIN
        
        rec_model_path = "./Paddle_OCR_weights/ppv3_rec.pth"
        dict_path = "./Paddle_OCR_weights/ppocr_keys_v1.txt"
        rec_config = AttrDict(
            in_channels=3,
            backbone=AttrDict(type='MobileNetV1Enhance', scale=0.5, last_conv_stride=[1, 2], last_pool_type='avg'),
            neck=AttrDict(type='None'),
            head=AttrDict(type='Multi', head_list=AttrDict(
                CTC=AttrDict(Neck=AttrDict(name="svtr", dims=64, depth=2, hidden_dims=120, use_guide=True)),
                # SARHead=AttrDict(enc_dim=512,max_text_length=70)
                ), n_class=6625) )
    
        rec_model = RecModel(rec_config)
        rec_model.load_state_dict(torch.load(rec_model_path))
        if(use_cuda):
            rec_model.to(self.device)
            rec_model = torch.nn.DataParallel(rec_model, device_ids=range(cfg.ngpu))
        
        
        #rec_model.eval()
        #print(rec_model)
        print("Paddle_OCR read ok ~~~~~~~~~~~~~~~~~~~~~~~~")
        return rec_model
        
    def CRNN_init(self, recognizer_path=None, opt=None):
        

        
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        #print("CRNNN",model)
        macs, params = ptflops.get_model_complexity_info(model, (1, 32, 100), as_strings=True,
                                                          print_per_layer_stat=False, verbose=True) #print_per_layer_stat=True
        #print("---------------- TP Module -----------------")
        #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        #print("--------------------------------------------")
        
        recognizer_path = 'crnn.pth'#'./ckpt/vis_TPGSR-TSRN/recognizer_best_0.pth'
        
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        # print("stat_dict:", stat_dict.keys())
        # if recognizer_path is None:
        model.load_state_dict(stat_dict)
        try:
            model.load_state_dict(stat_dict)
            print("crnn load weight ok ")
        # else:
        except Exception:
            model = stat_dict
            print("crnn load weight error")
        # model #.eval()
        # model.eval()

        return model, aster_info
        
        
        
    def CRNNRes18_init(self, recognizer_path=None, opt=None):
        model = crnn.CRNN_ResNet18(32, 1, 37, 256)
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        # print("stat_dict:", stat_dict.keys())
        # if recognizer_path is None:
        #     if stat_dict == model.state_dict():
        #         model.load_state_dict(stat_dict)
        # else:
        #     model = stat_dict

        # model #.eval()
        # model.eval()
        return model, aster_info

    def TPG_init(self, recognizer_path=None, opt=None):
        model = crnn.Model(opt)

        macs, params = ptflops.get_model_complexity_info(model, (1, 32, 100), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True) #print_per_layer_stat=True
        
        #print("---------------- TP Module -----------------")
        #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        #print("--------------------------------------------")
        
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else opt.saved_model
        print('loading pretrained TPG model from %s' % model_path)
        stat_dict = torch.load(model_path)

        model_keys = model.state_dict().keys()
        #print("state_dict:", len(stat_dict))
        #if type(stat_dict) == list:
        #    print("state_dict:", len(stat_dict))
        #    stat_dict = stat_dict[0]#.state_dict()
        #load_keys = stat_dict.keys()

        # if recognizer_path is None:
        try:
            # model.load_state_dict(stat_dict)
            load_keys = stat_dict.keys()
            man_load_dict = model.state_dict()
            for key in stat_dict:
                if not key.replace("module.", "") in man_load_dict:
                    print("Key not match", key, key.replace("module.", ""))
                man_load_dict[key.replace("module.", "")] = stat_dict[key]
            model.load_state_dict(man_load_dict)
        except Exception:
            model = stat_dict

        return model, aster_info

    def parse_SEED_data(self, inputs):
        input_dict = {}
        # if global_args.evaluate_with_lexicon:
        # imgs, label_encs, lengths, file_name = inputs
        # else:
        #     # imgs, label_encs, lengths = inputs
        #     imgs, label_encs, lengths, embeds_ = inputs

        imgs_input = inputs

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100

        if self.args.random_reso:
            batch_size = len(imgs_input)
            new_input = []
            for img in imgs_input:
                new_input.append(torch.nn.functional.interpolate(img[:, :3, ...], (32, in_width), mode='bicubic'))
            imgs_input = torch.cat(new_input, 0)
        else:
            batch_size = imgs_input.shape[0]
            imgs_input = torch.nn.functional.interpolate(imgs_input[:, :3, ...], (32, in_width), mode='bicubic')

        label_encs, lengths, file_namem, embeds_ = None, [25 for i in range(batch_size)], None, None

        with torch.no_grad():
            images = imgs_input.to(self.device).sub_(0.5).div_(0.5)
            if label_encs is not None:
                label_encs = label_encs.to(self.device)
            if embeds_ is not None:
                embeds_ = embeds_.to(self.device)
        input_dict['images'] = images
        input_dict['rec_targets'] = label_encs
        input_dict['rec_lengths'] = lengths
        input_dict['rec_embeds'] = embeds_
        # if global_args.evaluate_with_lexicon:
        #     input_dict['file_name'] = file_name
        return input_dict

    def SEED_init(self, recognizer_path=None, opt=None):

        model = ModelBuilder(arch="ResNet_ASTER", rec_num_classes=94 + 3,
                       sDim=512, attDim=512, max_len_labels=100,
                       eos=94, STN_ON=True) #test_dataset.char2id[test_dataset.EOS]

        macs, params = ptflops.get_model_complexity_info(model, (3, 32, 100), as_strings=True,
                                                         print_per_layer_stat=True, verbose=True)
        print("---------------- TP Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")

        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else "./se_aster.tar"
        print('loading pretrained TPG model from %s' % model_path)
        # stat_dict = torch.load(model_path)

        model_keys = model.state_dict().keys()
        #print("state_dict:", len(stat_dict))
        #if type(stat_dict) == list:
        #    print("state_dict:", len(stat_dict))
        #    stat_dict = stat_dict[0]#.state_dict()
        #load_keys = stat_dict.keys()

        # if recognizer_path is None:
        '''
        try:
            # model.load_state_dict(stat_dict)
            load_keys = stat_dict.keys()
            man_load_dict = model.state_dict()
            for key in stat_dict:
                if not key.replace("module.", "") in man_load_dict:
                    print("Key not match", key, key.replace("module.", ""))
                man_load_dict[key.replace("module.", "")] = stat_dict[key]
            model.load_state_dict(man_load_dict)
        except Exception:
            model = stat_dict
        '''
        checkpoint = load_checkpoint(model_path)
        model.load_state_dict(checkpoint['state_dict'])

        model = model.to(self.device)

        return model, aster_info


    def parse_crnn_data(self, imgs_input):

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100
        
        if self.args.random_reso:
            batch_size = len(imgs_input)
            new_input = []
            for img in imgs_input:
                # print("img:", img.shape)
                if len(img.shape) < 4:
                    img = img.unsqueeze(0)
                new_input.append(torch.nn.functional.interpolate(img[:, :3, ...], (32, in_width), mode='bicubic'))
            imgs_input = torch.cat(new_input, 0)
        else:
            
            batch_size = imgs_input.shape[0]
            imgs_input = torch.nn.functional.interpolate(imgs_input[:, :3, ...], (32, in_width), mode='bicubic')

        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
        
        #imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
        
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def Aster_init(self):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TRAIN.VAL.rec_pretrained)['state_dict'])
        print('load pred_trained aster model from %s' % self.config.TRAIN.VAL.rec_pretrained)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        aster.eval()
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}

        if self.args.random_reso:
            batch_size = len(imgs_input)
            new_input = []
            for img in imgs_input:
                new_input.append(torch.nn.functional.interpolate(img[:, :3, ...], (32, 128), mode='bicubic'))
            imgs_input = torch.cat(new_input, 0)
        else:
            batch_size = imgs_input.shape[0]
            imgs_input = torch.nn.functional.interpolate(imgs_input[:, :3, ...], (32, 128), mode='bicubic')

        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)

# Text_Super-Resolution_from_TPGSR

# 程式架構&還原效果



# 復現產學程式碼

1. 下載[TPGSR](https://github.com/mjq11302010044/TPGSR)
2. 將 `main.py`  [](http://將main.py)`myself_text_dataloader.py`  放進(取代) ./TPGSR-main
3. 將 `base.py`  [](http://將main.py)`super_resolution.py`  放進(取代) ./TPGSR-main/interfaces
4. 將 `./Generate_text_blurred_images`  [](http://將main.py) `./Paddle_OCR_weights`  `./rec` [](http://將main.py)放進 ./TPGSR-main
5. 放自有的數據集 `./myself_data`  `./Text_Data`

(檔案太大，故改給路徑`./data/4TB/chen_hung/TPGSR-main/Text_Data` )

(在`base.py` 的 `get_train_myself_dat_aug_case1_multisize` 內 `myself_text_datasets_for_aug_collate` 之 參數`root_HR` 改成上面的路徑)

1. 將 `./model/tsrn.py`  將`TSRN_TL`的`text_emb`改成6625
2. 將 `super_resolution.yaml`    放進(取代) ./TPGSR-main/config

訓練指令

```python
python3 main.py --arch="tsrn_tl_cascade" --batch_size=8 --val_batch_size=32 --mask --use_distill --gradient --sr_share --stu_iter=1 --vis_dir='train_part1'  --use_paddleocr_val --myself_data --use_paddleocr --use_myself_aug --use_padding --aug_prob=0.9 
```

詳細說明:

```python
--vis_dir #存放的資料夾
--aug_prob #數據增強的機率
--myself_data #是否使用自有的數據集做訓練
--use_myself_aug #是否使用自有的數據增強方法
--use_paddleocr_val #驗證時採用paddleocr來作文本識別
--use_paddleocr #訓練時使用paddleocr取文本特徵
--use_padding #多尺度訓練中，使用padding；否則是resize
```

Demo指令:

```python
python3 main.py --arch="tsrn_tl_cascade" --test_model="CRNN" --batch_size=1 --sr_share --gradient --demo --stu_iter=1 --vis_dir='./display/LR_b5_multi_pad_no_refine' --mask --resume='./ckpt/vis_TPGSR-TSRN_paddle_parse_resize_myself_display_aug_no_pretrain_multisize_padding_valbs/model_best_0_11_48800_20.pth' --demo_dir='./test_datasets_demo/LR_b5'
```

```python
python3 main.py --arch="tsrn_tl_cascade" --test_model="CRNN" --batch_size=1 --sr_share --gradient --demo --stu_iter=1 --vis_dir='./display/test_real' --mask --resume='./ckpt/train_add_redstamp_new_val_resize/model_best_0_4_15400_18.pth' --demo_dir='./demo_img/real_image'
```

詳細說明:

```python
--demo_dir #待還原的圖片
--resume #讀取的權重
--vis_dir #輸出還原結果，和訓練時的差別在於要多加個./display
```

`./ckpt`權重ckpt存檔規則:

```yaml
範例: model_best_0_56_360000_12.pth

說明: model_best_{第幾個模型，都是0，所以不太重要}_{第幾個epoch}_{第幾次迭代}_{驗證集的文本辨識準確率}.pth
```

`./tensorboard` 可視化結果觀察

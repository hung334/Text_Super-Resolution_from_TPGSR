import os
import shutil
import sys

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import random

def generate_stamp(name):
    
    
    #wd = os.getcwd()
    #print("working directory is ", wd)
    
    blank = "    "
    image = Image.new('RGB', (650, 650), "white")

    # 获取一个画笔对象，将图片对象传过去
    draw = ImageDraw.Draw(image)

    # 获取一个font字体对象参数是ttf的字体文件的目录，以及字体的大小
    #font = ImageFont.truetype("./fonts/msyh.ttf", size=104)
    font = ImageFont.truetype("./Generate_Stamp/fonts/msyh.ttf", size=104)

    # 在图片上写东西,参数是：定位，字符串，颜色，字体
    draw.ellipse([(20, 20), (630, 630)], fill=None, outline='red', width=12)
    firstname = name.get("first_name")
    centername = name.get("center_name")
    lastname = name.get("last_name")

    if firstname == '合' and lastname == '格':
        draw.line([(40, 220), (610, 220)], fill='red', width=12)
        draw.line([(40, 415), (610, 415)], fill='red', width=12)

    if len(firstname) < 3:
        first_name_stamp = blank.join(firstname)
    elif len(firstname) == 3:
        first_name_stamp = " ".join(firstname)
    else:
        first_name_stamp = firstname

    if len(centername) < 3:
        center_name_stamp = blank.join(centername)
    elif len(centername) == 3:
        center_name_stamp = " ".join(centername)
    else:
        center_name_stamp = centername

    if len(lastname) < 3:
        last_name_stamp = blank.join(lastname)
    elif len(lastname) == 3:
        last_name_stamp = " ".join(lastname)
    else:
        last_name_stamp = lastname

    first_name_size = font.getsize(first_name_stamp)
    first_name_x = int(image.width / 2 - first_name_size[0] / 2)

    center_name_size = font.getsize(center_name_stamp)
    center_name_x = int(image.width / 2 - center_name_size[0] / 2)

    last_name_size = font.getsize(last_name_stamp)
    last_name_x = int(image.width / 2 - last_name_size[0] / 2)

    draw.text((first_name_x, 80), first_name_stamp, 'red', font=font, spacing=0, align='left')
    draw.text((center_name_x,random.randint(170,280)), center_name_stamp, 'red', font=font, spacing=0, align='left')
    draw.text((last_name_x, 410), last_name_stamp, 'red', font=font, spacing=0, align='left')

    #image.save(open('stamp/%s.png' % (firstname + lastname,), 'wb'), 'png', quality=100)
    return image


if __name__ == '__main__':
    
    
    firstname = '上海'
    centername = '證券'
    lastname = '公司'
    
    name = {'first_name': firstname,'center_name': centername,'last_name':lastname}
    print(name)
    
    stamp = generate_stamp(name)
    
    '''
    if os.path.exists("stamp"):
        shutil.rmtree("./stamp")
        os.mkdir("./stamp")
    else:
        os.mkdir("./stamp")
    name_list = []

    names_path = "names.txt"
    if not os.path.exists(names_path):
        print("没有找到 names.txt 文件")
        sys.exit(0)

    with open(names_path, 'r',encoding="utf-8") as f:
        line = f.readline()
        while line:
            name = line.strip().split('\t')
            name_list.append({
                'first_name': name[0],
                'last_name': name[1]
            })
            line = f.readline()
    for n in name_list:
        generate_stamp(n)
    '''
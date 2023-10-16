import json
import cv2
import numpy as np
from PIL import Image, ImageStat
from tqdm import tqdm
import os

def detect_color_image(file, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    pil_img = Image.open(file)
    bands = pil_img.getbands()
    if bands == ('R','G','B') or bands== ('R','G','B','A'):
        thumb = pil_img.resize((thumb_size,thumb_size))
        SSE, bias = 0, [0,0,0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias)/3 for b in bias ]
        for pixel in thumb.getdata():
            mu = sum(pixel)/3
            SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
        MSE = float(SSE)/(thumb_size*thumb_size)
        if MSE <= MSE_cutoff:
            return 1  # grayscale
        else:
            return 0  # color
        # print "( MSE=",MSE,")"
    elif len(bands)==1:
        return 1

def is_black_and_white(img_path, saturation_threshold=30):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    saturation = hsv_img[:,:,1]
    median_saturation = np.median(saturation)

    return median_saturation < saturation_threshold

img_folder = "/yuch_ws/views_release"

valid_path = "BLIP2_split_by_count.json"
with open(valid_path, 'r') as f:
    valid = json.load(f)

total = len(valid["-1"])
for i in range(14):
    total += len(valid[str(i)])
print("total is ", total)
print("for count ", -1 ," we have ", len( valid[str(-1)]) , ' samples ' ,len( valid[str(-1)])/total * 100, ' percentage.' )
for i in range(14):
    print("for count ", i ," we have ", len( valid[str(i)]) , ' samples ' ,len( valid[str(i)])/total * 100, ' percentage.' )

out_put_data = {}

for i in range(5,14):
    print("start section ", i)
    folders = valid[str(i)]
    out_put_data[str(i)] = {}
    out_put_data[str(i)]["grayscale"] = []
    out_put_data[str(i)]["color"] = []

    for j in tqdm(range(len(folders))):
        folder = folders[j]
        m1_count = 0
        m2_count = 0
        meta_path = img_folder + "/" + folder + "/objarverse_BLIP_metadata_v2.json"
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)

        for view in range(12):
            im_path = os.path.join(img_folder + "/" + folder, '%03d.png' % view)
            method1 = is_black_and_white(im_path, saturation_threshold=30)
            method2 = detect_color_image(im_path, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True)
            if method1:
                m1_count += 1
            if method2:
                m2_count += 1
        meta_data['m1_count'] = m1_count
        meta_data['m2_count'] = m2_count

        if m1_count >= 4 or m2_count >= 4:
            out_put_data[str(i)]["grayscale"].append(folder)
            meta_data['grayscale'] = True
        else:
            out_put_data[str(i)]["color"].append(folder)
            meta_data['grayscale'] = False

        new_meta_path = img_folder + "/" + folder + "/objarverse_BLIP_metadata_v3.json"
        with open(new_meta_path, 'w') as f:
            json.dump(meta_data,f)

out_path = "BLIP2_split_by_count_and_grayscale.json"
with open(out_path, 'w') as f:
    json.dump(out_put_data, f)

















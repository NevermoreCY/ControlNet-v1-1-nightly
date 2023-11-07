import cv2
import numpy as np
import json

file_path = 'shapenet_v1.json'
views_folder = "/yuch_ws/views_release"

with open(file_path, 'r') as f:
    folders = json.load(f)

c = 0
total = len(folders)
bad_path = []
good_path = []
for folder in folders:
    img_path = views_folder + '/' + folder + '/000.png'
    img =cv2.imread(img_path)

    if img.sum() == 328200:
        bad_path.append(folder)
        print(c , folder)
        c+=1
    else:
        good_path.append(folder)


out_path = 'shapenet_v1_good.json'
with open (out_path,'w') as f:
    json.dump(good_path,f)

out_path = 'shapenet_v1_bad.json'
with open (out_path,'w') as f:
    json.dump(bad_path,f)

print(c, total, c/total)
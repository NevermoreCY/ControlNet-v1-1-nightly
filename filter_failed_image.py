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
for folder in folders:
    img_path = views_folder + '/' + folder + '/000.png'
    img =cv2.imread(img_path)

    if img.sum() == 328200:
        bad_path.append(folder)
        print(c , folder)
        c+=1

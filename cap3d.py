import os

import pandas as pd

file = '/yuch_ws/3DGEN/cap3d/' + 'cap3d_objarverse_all.csv'
captions = pd.read_csv(file, header=None)

import json
import os


# shape_path = 'shapenet_v1_good.json'
# turbo_path = 'turbo_v1.json'
# turbo_scale = 1
# shape_scale = 1
#
#
# with open(shape_path, 'r') as f:
#     shape_data = json.load(f)
#
# with open(turbo_path, 'r') as f:
#     turbo_data = json.load(f)
#
# turbo_data = turbo_scale * turbo_data
# shape_data = shape_scale * shape_data
#
#
# print("turbo data final length: " , len(turbo_data))
# print("shape data final length: " , len(shape_data))


data = {}

cap3_data = {}
cap3_data[3] = []
for i in range(4,14):

    update_path = 'valid_paths_' + str(i) + '.json'
    with open (update_path,'r') as f:
        update_data = json.load(f)
    data[i] = update_data
    cap3_data[i] = []

c = 0
for item in captions[0]:
    c+=1
    print(c)
    for i in range(4,14):
        if item in data[i]:
            cap3_data[i].append(item)
        break
    cap3_data[3].append(item)


for i in range(4,14):
    print(i, len(cap3_data[i]))
out_path = 'cap3d_distribution.json'
with open(out_path,'w') as f:
    json.dump(out_path)







## if u want to obtain the caption for specific UID

#
#
#
# for item in captions:
#     print(item)
#     # print('cap3d:',captions[captions[0] == item][1].values[0])
#     data_path = '/home/nev/3DGEN/Results/12_11/cap3D/to_check/' + item + '/BLIP_best_text_v2.txt'
#     with open(data_path,'r')as f:
#         text = f.readline()
#     cap3d = captions[captions[0] == item][1].values[0]
#     print('cap3d:', cap3d)
#     print('Our cap:', text)
#
#     out_cap3d = '/home/nev/3DGEN/Results/12_11/cap3D/to_check/' + item + '/cap3d_' + cap3d
#     with open(out_cap3d,'w') as f:
#         f.write(cap3d)
#     out_our = '/home/nev/3DGEN/Results/12_11/cap3D/to_check/' + item + '/blip2_' + text
#     with open(out_our,'w') as f:
#         f.write(cap3d)

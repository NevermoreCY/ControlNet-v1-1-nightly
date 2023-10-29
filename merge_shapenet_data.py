import json
import os


shape_path = 'shapenet_v1.json'
with open(shape_path, 'r') as f:
    shape_data = json.load(f)

for i in range(8,14):

    update_path = 'valid_paths_' + str(i) + '.json'
    with open (update_path,'r') as f:
        update_data = json.load(f)

    update_data = shape_data + update_data

    out_path = 'valid_shape_paths_' + str(i) + '.json'
    with open(out_path,'w') as f:
        json.dump(update_data, f)





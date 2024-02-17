# import numpy as np
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
import os
import json
#
for i in range(32):
    name = 'views_subdir' + str(i)
    path = '/yuch_ws/views_valid5/objaverse/control3D/' + name + '/*'
    cmd = 'mv ' + name + ' views'
    print(cmd)
    os.system(cmd)

# L = os.listdir('mscoco/mscoco')
#
#
# name_set = set([])
#
# for item in L:
#     L = item.split('.')
#     idx = L[0]
#     if len(idx) == 9:
#         name_set.add(idx)
#
# name_list = list(name_set)
# print(len(name_list) , max(name_list))
#
# out_path = 'mscoco_name_list.json'
#
# with open(out_path,'w') as f:
#     json.dump(name_list,f)





















import json
from tqdm import tqdm
# total_jobs= 38
#
# count_dict= {}
# for i in range(14):
#     count_dict[str(i)] = []
#
# for i in range(total_jobs):
#     file_name = "BLIP_v2_data_split_" + str(i) + ".json"
#     with open(file_name, 'r') as f:
#         cur_json = json.load(f)
#     for key in cur_json:
#         # print(type(key))
#         count_dict[key].extend( cur_json[key] )
#         print(i, key, len(cur_json[key]))
#
# for i in range(14):
#     print("for count ", i ," we have ", len( count_dict[str(i)]) , ' samples ')
#
# with open("BLIP2_split_by_count.json", 'w') as f:
#     json.dump(count_dict, f)
#

valid_path = "valid_paths.json"

out = {}
for i in range(14):
    out[i] = []

with open(valid_path, 'r') as f:
    valid = json.load(f)

prefix = "/yuch_ws/views_release/"
for i in tqdm(range(len(valid))):
    folder = valid[i]
    json_path = prefix + folder + "/objarverse_BLIP_metadata_v2.json"
    with open(json_path, 'r') as f:
        meta = json.load(json_path)
    out[meta['count']].append(meta['Best_text'])

for i in range(14):
    print("for count ", i ," we have ", len( out[i]) , ' samples ')

with open("BLIP2_split_by_count.json", 'w') as f:
    json.dump(out, f)


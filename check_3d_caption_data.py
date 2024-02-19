import json
import pandas as pd


file = 'Cap3D_automated_Objaverse_full_no3Dword.csv'
captions = pd.read_csv(file, header=None)

cap_data = {}

for i in range(len(captions)):
    str_id = captions[0][i]
    str_cap = captions[1][i]
    cap_data[str_id] = str_cap

our_data = 'valid_paths_5.json'

with open(our_data,'r') as f:
    our_data = json.load(f)




bad_files = []
to_save = []
cap_key_list = list(cap_data.keys())

c = 0
for key in our_data
    c+=1
    if c % 1000:
        print(c/len(our_data))
    if key not in cap_key_list:
        bad_files.append(key)
        print(key )
    else:
        to_save.append(key)

print(len(bad_files))
print(len(to_save))

out_data = 'valid_paths_5_v2.json'

with open(out_data,'w') as f:
    json.dump(to_save,f)




















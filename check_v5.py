import json

file = 'BLIP2_split_by_count_recheck_tag_V5.json'

with open (file,'r') as f:
    x = json.load(f)

for i in range(14):
    print(i, len(x[str(i)]))
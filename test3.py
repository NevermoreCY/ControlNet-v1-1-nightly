import json


merged_data = {}

for i in range(5,14):
    file_pre = 'BLIP2_split_by_count_and_grayscale' + str(i) + '.json'
    with open(file_pre, 'r') as f:
        data = json.load(f)
    merged_data[str(i)] = data[str(i)]
    data = data[str(i)]

    print('For count ', str(i) , '  grayscale ', len(data[' grayscale']) , ' color count', len(data[' color']))

out_file = 'BLIP2_split_by_count_and_grayscale.json'
with open(out_file, 'w') as f:
    json.dump(merged_data, f)


data2 = 'BLIP2_split_by_count_recheck_tag_V4.json'
with open(data2, 'r') as f:
    valid = json.load(f)
total= 0
for i in range(14):
    total += len(valid[str(i)])
print("total is ", total)
print("for count ", -1 ," we have ", len( valid[str(-1)]) , ' samples ' ,len( valid[str(-1)])/total * 100, ' percentage.' )
for i in range(14):
    print("for count ", i ," we have ", len( valid[str(i)]) , ' samples ' ,len( valid[str(i)])/total * 100, ' percentage.' )
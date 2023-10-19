import json


for i in range(5,14):
    file_pre = 'BLIP2_split_by_count_and_grayscale' + str(i) + '.json'
    with open(file_pre, 'r') as f:
        data = json.load(f)
    print(data.keys())
    print('For count ', str(i) , 'grayscale ', len(data['grayscale']) , 'color count', len(data['color']))

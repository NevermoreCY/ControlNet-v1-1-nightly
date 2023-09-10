import os

from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import json
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cwd = os.getcwd()
print("cwd is ", cwd)
os.chdir("BLIP")
cwd = os.getcwd()
print("cwd is ", cwd)
print("device is ", device )

def load_demo_image(image_size, device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    w, h = raw_image.size
    #display(raw_image.resize((w // 5, h // 5)))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image
def load_image(image_size, device, im_path):

    raw_image = Image.open(im_path).convert('RGB')
    # print("raw_image shape", raw_image.size)
    # print("raw_image type", type(raw_image))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    # image = transform(raw_image).unsqueeze(0).to(device)
    image = transform(raw_image).to(device)
    return image

def load_image_batch(image_size, device, batch_path):

    raw_image = Image.open(im_path).convert('RGB')
    # print("raw_image shape", raw_image.size)
    # print("raw_image type", type(raw_image))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

from BLIP.models.blip import blip_decoder



model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
image_size = 256

model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)


cwd = os.getcwd()
print("cwd is ", cwd)
os.chdir("../")
cwd = os.getcwd()
print("cwd is ", cwd)


def most_frequent(List):
    counter = 0
    item = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            item = i
    return item

img_folder = "/yuch_ws/views_release"
# sub_folder_list = os.listdir(img_folder)
# sub_folder_list.sort()

with open('valid_paths.json') as f:
    sub_folder_list = json.load(f)

sub_folder_list.sort()



total_n = len(sub_folder_list)
print("total_n", total_n)  # 772870
print("first few names",sub_folder_list[:5])

job_num = 0
job_length = total_n // 8

print("******** cur job_num is " , job_num)
start_n = 10000 * job_num
end_n = 10000 * job_num + 1
bz = 10

batch_s = start_n
batch_e = batch_s + bz

while batch_s < end_n:
    print(batch_s, batch_e)
    iter_time_s = time.time()

    batch_names =  sub_folder_list[batch_s:batch_e]
    images = []


    curr = time.time()
    # print("time load_image", curr)
    for j in range(bz):
        print(j)
        folder = batch_names[j]
        if folder[-4:] != "json":
            for i in range(12):
                im_path = os.path.join(img_folder + "/" + folder, '%03d.png' % i)
                images.append(load_image(image_size=image_size, device=device, im_path=im_path))
    #print(assert(bz*12 == len(images)) )
    next_t = time.time()
    # print(" time after load_image =", next_t)
    print("time for load diff 1", next_t - curr)

    print("total num is ", len(images), ", should be",bz*12)

    # make them a batch
    batch_images = torch.stack(images, 0)
    print("batch shape is ", batch_images.shape)
    with torch.no_grad():
        # beam search
        curr = time.time()
        # print("time before inference", curr)
        captions = model.generate(batch_images, sample=False, num_beams=3, max_length=20, min_length=5)

        next_t = time.time()
        # print(" time after inference =", next_t)
        print("time for inference diff 2", next_t - curr)
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        #        print('caption: ' + caption[0])


    # post process
    curr = time.time()
    # print("time before post", curr)
    print("num of captions is ", len(captions) , "should be ", bz* 12 )
    for j in range(bz):
        folder = batch_names[j]
        cur_texts = captions[j*12:(j+1)*12]
        print(len(cur_texts))
        best_text = most_frequent(cur_texts)
        out_text_name = img_folder + "/" + folder + "/BLIP_best_text.txt"
        with open(out_text_name, 'w') as f:
            f.write(best_text)
    # print(" time after post =", next_t)
    print("time for post diff 3", next_t - curr)

    # update id
    batch_s += bz
    batch_e += bz
    time_cost = time.time() - iter_time_s
    print("1 iteration takes time :", time_cost /60 , " minutes." )
    # for f_id in tqdm(range(len(sub_folder_list))):
    #     folder = sub_folder_list[f_id]
    #     if folder[-4:] != "json":
    #         texts = []
    #         for i in range(12):
    #             im_path = os.path.join(img_folder + "/" + folder, '%03d.png' % i)
    #
    #             curr = time.time()
    #             print("time load_image", curr)
    #             x = load_image(image_size=image_size, device=device, im_path=im_path)
    #             next_t = time.time()
    #             print(" time after load_image =", next_t)
    #             print("diff 1", next_t - curr)
    #
    #             image = x.repeat(1800, 1, 1, 1)
    #             print(image.shape)
    #
    #
    #             with torch.no_grad():
    #                 # beam search
    #                 curr = time.time()
    #                 print("time before inference", curr)
    #                 caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
    #                 next_t = time.time()
    #                 print(" time after inference =", next_t)
    #                 print("diff 2", next_t - curr)
    #                 # nucleus sampling
    #                 # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
    #         #        print('caption: ' + caption[0])
    #                 texts.append(caption[0])
    #
    #         out_text_name = img_folder + "/" + folder + "/BLIP_text.txt"
    #         name = most_frequent(texts)
    #
    #         with open(out_text_name, 'w') as f:
    #             f.write(name)
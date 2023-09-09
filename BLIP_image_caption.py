import os

from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cwd = os.getcwd()
print("cwd is ", cwd)
os.chdir("BLIP")
cwd = os.getcwd()
print("cwd is ", cwd)

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
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

from BLIP.models.blip import blip_decoder



model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
image_size = 512

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
sub_folder_list = os.listdir(img_folder)
sub_folder_list.sort()

cur_n = 0
total_n = len(sub_folder_list)
print(sub_folder_list[:5])
for f_id in tqdm(range(len(sub_folder_list))):
    folder = sub_folder_list[f_id]
    if folder[-4:] != "json":
        texts = []
        for i in range(12):
            im_path = os.path.join(img_folder + "/" + folder, '%03d.png' % i)


            image = load_image(image_size=image_size, device=device, im_path=im_path)

            with torch.no_grad():
                # beam search
                caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
                # nucleus sampling
                # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        #        print('caption: ' + caption[0])
                texts.append(caption[0])

        out_text_name = img_folder + "/" + folder + "/BLIP_text.txt"
        name = most_frequent(texts)

        with open(out_text_name, 'w') as f:
            f.write(name)
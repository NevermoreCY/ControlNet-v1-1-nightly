import torch
from PIL import Image

test_folder = 'test/turbo_cases/'
test_img = test_folder + 'a black Donkey.png'
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open(test_img).convert("RGB")
# display(raw_image.resize((596, 437)))

import torch
from lavis.models import load_model_and_preprocess
# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

print("Done preparing")

model.generate({"image": image, "prompt": "Question: This is a rendering image of a 3D asset, can you generate a caption for it with details such as high polygon or low polygon, action, facing and style? Answer:"})


model.generate({"image": image, "prompt": "Question: This is a rendering image of a 3D asset, can you generate a caption for it? Answer:"})


import os

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

img_names = os.listdir(test_folder)

print("Done preparing")

for img in img_names:

    print("** For image : ", img)
    print('Single question without remembering previous context:')
    raw_image = Image.open(test_folder+img).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    q1a= model.generate({"image": image, "prompt": "Question: Can you generate a caption for this image as detail as possible. Including the object's facing direction, color, action and style. please ignore the black background. Answer:"})
    print(q1a)
    q1b= model.generate({"image": image, "prompt": "Question: Can you generate a caption for this image as detail as possible. please ignore the black background. Answer:"})
    print(q1b)
    q2 = model.generate({"image": image, "prompt": "Question: This is a rendering image of a 3D asset, can you tell me whether it is high poly or low poly? Answer:"})
    print(q2)
    q3 = model.generate({"image": image, "prompt": "Question: Can you tell me which direction is it facing? Answer:"})
    print(q3)
    q4 = model.generate({"image": image, "prompt": "Question: can you tell me what action is this object doing? please ignore the black background Answer:"})
    print(q4)
    q5 = model.generate({"image": image, "prompt": "Question: can you tell me the style of this object? Answer:"})
    print(q5)
    q6 = model.generate({"image": image, "prompt": "Question: This is a rendering image of a 3D asset, can you tell me whether the object has texture? Answer:"})
    print(q6)

    print('Ask Question with context:')
    cur_prompt = "Question: Can you generate a caption for this image as detail as possible. Please ignore the black background. Answer:"
    answer = model.generate({"image": image, "prompt": cur_prompt})
    print(cur_prompt, answer)
    Q = 'Can you tell me which direction is it facing?'
    cur_prompt = 'Question: '+ Q + ' Answer:'
    answer = model.generate({"image": image, "prompt": cur_prompt})
    print(Q,answer)

    Q = 'Can you tell me what action is it doing? Please ignore the black background.'
    cur_prompt = 'Question: '+ Q + ' Answer:'
    answer = model.generate({"image": image, "prompt": cur_prompt})
    print(Q,answer)

    Q = 'Can you tell me the style of this image? '
    cur_prompt = 'Question: '+ Q + ' Answer:'
    answer = model.generate({"image": image, "prompt": cur_prompt})
    print(Q,answer)

    Q = 'This is a rendering image of a 3D asset, Can you tell me whether it is high poly or low poly? '
    cur_prompt = 'Question: '+ Q + ' Answer:'
    answer = model.generate({"image": image, "prompt": cur_prompt})
    print(Q,answer)

    Q = 'Can you tell me whether the object has texture or not? '
    cur_prompt = 'Question: '+ Q + ' Answer:'
    answer = model.generate({"image": image, "prompt": cur_prompt})
    print(Q,answer)

print("Done inferencing")


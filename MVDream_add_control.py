
# assert len(sys.argv) == 3, 'Args are wrong.'

input_path_control = 'base_models/control_sd15_canny.pth'
input_path_control2 = 'base_models/control_v11p_sd15_canny.pth'
input_path_mvd = 'base_models/sd-v1.5-4view.pt'
output_path = 'base_models/mvcontrol_base_v5.pt'

# assert os.path.exists(input_path_control), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
# assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
# from share import *
from cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

# new model (MVDream + control)
model = create_model(config_path='./base_models/control_3D_sd15.yaml')
# model = create_model(config_path='./models/yuch_v11p_sd15_canny_full.yaml')

print("model creation done!")

# load pretrained A
pretrained_weights_control = torch.load(input_path_control)

if 'state_dict' in pretrained_weights_control:
    pretrained_weights_control = pretrained_weights_control['state_dict']


# load pretrained B

pretrained_weights_control2 = torch.load(input_path_control2)

if 'state_dict' in pretrained_weights_control2:
    pretrained_weights_control2 = pretrained_weights_control2['state_dict']

# load c

pretrained_weights_mvd = torch.load(input_path_mvd)

if 'state_dict' in pretrained_weights_mvd:
    pretrained_weights_mvd = pretrained_weights_mvd['state_dict']


control_key = list(pretrained_weights_control.keys())
control_key2 = list(pretrained_weights_control2.keys()) # note all item in control_key2 are also included in control_key
mvd_key = list(pretrained_weights_mvd.keys())
# print('\n\n\n\n\n',pretrained_weights_control.keys())
# print('\n\n\n\n\n',pretrained_weights_control2.keys())
# print('\n\n\n\n\n',pretrained_weights_mvd.keys())


# print('\n\n\n\n\n\n  in control key 2, not in control key')
# for item in control_key2:
#     if item not in control_key:
#         print(item) # None

control3D_dict = model.state_dict()
control3D_key = list(control3D_dict.keys())

print(len(control_key) , len(control_key2), len(mvd_key), len(control3D_dict))


print('\n\n\n\n\n\n  in mvd, not in con3d')
for item in mvd_key:
    print(item, pretrained_weights_mvd[item].shape)

print('\n\n\n\n\n\n  in control3d, not in mvd')
for item in control3D_key:
    if item not in mvd_key:
        print(item, control3D_dict[item].shape)

print('\n\n\n\n\n\n  in control3d, not in control')
for item in control3D_key:
    if item not in control_key:
        print(item, control3D_dict[item].shape )

print('\n\n\n\n\n\n  in mvd, not in con3d')
for item in mvd_key:
    if item not in control3D_key:
        print(item, pretrained_weights_mvd[item].shape)


print('\n\n\n\n\n\n  in con, not in con3d')
for item in control_key:
    if item not in control3D_key:
        print(item, pretrained_weights_control[item].shape)




target_dict = {}
# 0th step copy original weights, these are all the keys we nedd
for k in control3D_dict.keys():
    # print("control 3D " , k )
    target_dict[k] = control3D_dict[k].clone()
    # target_dict[k].requires_grad = True
# First copy control net v1.0 parameters
for k in pretrained_weights_control.keys():
    # print("control net v1  ", k)
    target_dict[k] = pretrained_weights_control[k].clone()
    # target_dict[k].requires_grad = True
# second copy control net v1.1 parameters
for k in pretrained_weights_control2.keys():
    # print("control net v2  ", k)
    target_dict[k] = pretrained_weights_control2[k].clone()
    # target_dict[k].requires_grad = True

# copy mvd
for k in pretrained_weights_mvd.keys():

    if ('model.diffusion_model.time_embed.' in k):
        print("time in MVD!, copy it")
        prefix_l = len('model.diffusion_model.time_embed.')
        sufix = k[prefix_l:]
        print('sufix:', sufix)
        target_pre = 'control_model.time_embed.'
        target_key = target_pre + sufix
        print("TO : " , target_key)
        target_dict[target_key] = pretrained_weights_mvd[k].clone()
        # target_dict[target_key].requires_grad = True
    elif ('model.diffusion_model.camera_embed.' in k):
        print("camera in MVD!, copy it")
        prefix_l = len('model.diffusion_model.camera_embed.')
        sufix = k[prefix_l:]
        print('sufix:', sufix)
        target_pre = 'control_model.camera_embed.'
        target_key = target_pre + sufix
        print("TO : ", target_key)
        target_dict[target_key] = pretrained_weights_mvd[k].clone()
        # target_dict[target_key].requires_grad = False
    elif ('model.diffusion_model.input_blocks.' in k):
        print(" copy input block from ", k)
        prefix_l = len('model.diffusion_model.input_blocks.')
        sufix = k[prefix_l:]
        print('sufix:', sufix)
        target_pre = 'control_model.input_blocks.'
        target_key = target_pre + sufix
        print("TO : ", target_key)
        target_dict[target_key] = pretrained_weights_mvd[k].clone()
        # target_dict[target_key].requires_grad = True
    elif ('model.diffusion_model.middle_block.' in k):
        print("copy middle block from ", k)
        prefix_l = len('model.diffusion_model.middle_block.')
        sufix = k[prefix_l:]
        print('sufix:', sufix)
        target_pre = 'control_model.middle_block.'
        target_key = target_pre + sufix
        print("TO : ", target_key)
        target_dict[target_key] = pretrained_weights_mvd[k].clone()
        # target_dict[target_key].requires_grad = True
    else:
        print("\n ELSE case for ", k )
        if 'cond_stage_model' not in k:
            target_dict[k] = pretrained_weights_mvd[k].clone()
            # target_dict[k].requires_grad = True

# for k in pretrained_weights_mvd.keys():


to_discard = ["model.diffusion_model.time_embed.0.weight", "model.diffusion_model.time_embed.0.bias", "model.diffusion_model.time_embed.2.weight", "model.diffusion_model.time_embed.2.bias"]
for k in to_discard:
    target_dict.pop(k,None)


for key in target_dict:
    item = target_dict[key]
    print(key)

    if 'control_model' in key and 'camera' not in key:
        target_dict[key].requires_grad = True



    if 'cond_stage_model' not in key:
        # target_dict[key].requires_grad = True
        print(type(item))
        print(item.requires_grad)


# model.load_state_dict(target_dict, strict=True)
# torch.save(model.state_dict(), output_path)
# print('Done.')


#
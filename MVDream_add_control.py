import sys
import os

# assert len(sys.argv) == 3, 'Args are wrong.'

input_path_control = 'base_models/control_sd15_canny.pth'
input_path_control2 = 'base_models/control_v11p_sd15_canny.pth'
input_path_mvd = 'base_models/sd-v1.5-4view.pt'
output_path = 'base_models/mvcontrol_base.pt'

# assert os.path.exists(input_path_control), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
# assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import *
from cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

# new model (MVDream + control)
model = create_model(config_path='./models/control_3D_sd15.yaml')

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

print(len(control_key) , len(control_key2), len(mvd_key))

# print('\n\n\n\n\n\n  in mvd key, not in control key')
# for item in mvd_key:
#     if item not in control_key:
#         print(item)
# model.diffusion_model.camera_embed.0.weight
# model.diffusion_model.camera_embed.0.bias
# model.diffusion_model.camera_embed.2.weight
# model.diffusion_model.camera_embed.2.bias

# for item in control_key:
#     if item not in mvd_key:
#         print(item)

    # control_model.xxx


#
# scratch_dict = model.state_dict()
#
# target_dict = {}
# for k in scratch_dict.keys():
#     is_control, name = get_node_name(k, 'control_')
#     if is_control:
#         copy_k = 'model.diffusion_' + name
#     else:
#         copy_k = k
#     if copy_k in pretrained_weights:
#         target_dict[k] = pretrained_weights[copy_k].clone()
#     else:
#         target_dict[k] = scratch_dict[k].clone()
#         print(f'These weights are newly added: {k}')
#
# model.load_state_dict(target_dict, strict=True)
# torch.save(model.state_dict(), output_path)
# print('Done.')

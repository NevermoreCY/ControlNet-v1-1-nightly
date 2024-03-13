
trained_weight_path = 'logs/2024-03-12T02-41-07_MVD_cfg9_control3D_trainv4/checkpoints/epoch=9-step=15599.ckpt'
# pretrained_weight_path = 'base_models/mvcontrol_base_v4.pt'
pretrained_weight_path = 'logs/2024-03-13T04-23-06_MVD_no_res_3d_cfg9_control3D_trainv4/checkpoints/epoch=0-step=2399.ckpt'

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



cuda0 = torch.device('cuda:0')
# load trained B

pretrained_weights = torch.load(pretrained_weight_path)
print(type(pretrained_weights) )

if 'state_dict' in pretrained_weights:
    print(1)
    pretrained_weights = pretrained_weights['state_dict']

# load c

trained_weights = torch.load(trained_weight_path)
print(type(trained_weights), print(trained_weights.keys()))

if 'state_dict' in trained_weights:
    print(2)
    trained_weights = trained_weights['state_dict']
#
# print(type(pretrained_weights))
#
# print(type(trained_weights))
# pretrained_weights.to(cuda0)
# trained_weights.to(cuda0)



pretrained_key = list(pretrained_weights.keys())
trained_key = list(trained_weights.keys()) # note all item in control_key2 are also included in control_key

print("loading done ! ")

for item in pretrained_key:

    pre_item = pretrained_weights[item]
    trained_item = trained_weights[item]

    # diff = pre_item - trained_item
    # x = torch.sum(diff)
    item_shape = pre_item.shape
    totral_param = 1
    for num in item_shape:
        totral_param *= num
    # print(item , pre_item.shape )
    # print("\n" , totral_param,x )
    print(pre_item, trained_item)

    # print(item, pretrained_weights_mvd[item].shape)
#
# print('\n\n\n\n\n\n  in control3d, not in mvd')
# for item in control3D_key:
#     if item not in mvd_key:
#         print(item, control3D_dict[item].shape)
#
# print('\n\n\n\n\n\n  in control3d, not in control')
# for item in control3D_key:
#     if item not in control_key:
#         print(item, control3D_dict[item].shape )
#
# print('\n\n\n\n\n\n  in mvd, not in con3d')
# for item in mvd_key:
#     if item not in control3D_key:
#         print(item, pretrained_weights_mvd[item].shape)
#
#
# print('\n\n\n\n\n\n  in con, not in con3d')
# for item in control_key:
#     if item not in control3D_key:
#         print(item, pretrained_weights_control[item].shape)
#
#
#
#
# target_dict = {}
# # 0th step copy original weights, these are all the keys we nedd
# for k in control3D_dict.keys():
#     target_dict[k] = control3D_dict[k].clone()
# # First copy control net v1.0 parameters
# for k in pretrained_weights_control.keys():
#     target_dict[k] = pretrained_weights_control[k].clone()
# # second copy control net v1.1 parameters
# for k in pretrained_weights_control2.keys():
#     target_dict[k] = pretrained_weights_control2[k].clone()
#
# # copy mvd
# for k in pretrained_weights_mvd.keys():
#
#     if ('model.diffusion_model.time_embed.' in k):
#         print("time in MVD!, copy it")
#         prefix_l = len('model.diffusion_model.time_embed.')
#         sufix = k[prefix_l:]
#         print('sufix:', sufix)
#         target_pre = 'control_model.time_embed.'
#         target_key = target_pre + sufix
#         print("TO : " , target_key)
#         target_dict[target_key] = pretrained_weights_mvd[k].clone()
#     elif ('model.diffusion_model.camera_embed.' in k):
#         print("camera in MVD!, copy it")
#         prefix_l = len('model.diffusion_model.camera_embed.')
#         sufix = k[prefix_l:]
#         print('sufix:', sufix)
#         target_pre = 'control_model.camera_embed.'
#         target_key = target_pre + sufix
#         print("TO : ", target_key)
#         target_dict[target_key] = pretrained_weights_mvd[k].clone()
#     elif ('model.diffusion_model.input_blocks.' in k):
#         print(" copy input block from ", k)
#         prefix_l = len('model.diffusion_model.input_blocks.')
#         sufix = k[prefix_l:]
#         print('sufix:', sufix)
#         target_pre = 'control_model.input_blocks.'
#         target_key = target_pre + sufix
#         print("TO : ", target_key)
#         target_dict[target_key] = pretrained_weights_mvd[k].clone()
#     elif ('model.diffusion_model.middle_blocks.' in k):
#         print("copy middle block from ", k)
#         prefix_l = len('model.diffusion_model.middle_blocks.')
#         sufix = k[prefix_l:]
#         print('sufix:', sufix)
#         target_pre = 'control_model.middle_blocks.'
#         target_key = target_pre + sufix
#         print("TO : ", target_key)
#         target_dict[target_key] = pretrained_weights_mvd[k].clone()
#     else:
#         target_dict[k] = pretrained_weights_mvd[k].clone()
#
# # for k in pretrained_weights_mvd.keys():
#
#
# to_discard = ["model.diffusion_model.time_embed.0.weight", "model.diffusion_model.time_embed.0.bias", "model.diffusion_model.time_embed.2.weight", "model.diffusion_model.time_embed.2.bias"]
# for k in to_discard:
#     target_dict.pop(k,None)
#
#
# model.load_state_dict(target_dict, strict=True)
# torch.save(model.state_dict(), output_path)
# print('Done.')


#
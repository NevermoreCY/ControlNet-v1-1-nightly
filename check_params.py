
# assert len(sys.argv) == 3, 'Args are wrong.'


m_path = 'base_models/mvcontrol_base_v2.pt'

# assert os.path.exists(input_path_control), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
# assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
# from share import *
from cldm.model import create_model



# new model (MVDream + control)
# model = create_model(config_path='./base_models/control_3D_sd15.yaml')
# # model = create_model(config_path='./models/yuch_v11p_sd15_canny_full.yaml')
#
# print("model creation done!")

# load pretrained A
pretrained_weights_control = torch.load(m_path)

if 'state_dict' in pretrained_weights_control:
    pretrained_weights_control = pretrained_weights_control['state_dict']


# load pretrained B

control_key = list(pretrained_weights_control.keys())

# print('\n\n\n\n\n',pretrained_weights_control.keys())
# print('\n\n\n\n\n',pretrained_weights_control2.keys())
# print('\n\n\n\n\n',pretrained_weights_mvd.keys())


# print('\n\n\n\n\n\n  in control key 2, not in control key')
# for item in control_key2:
#     if item not in control_key:
#         print(item) # None

print('\n\n\n\n\n\n  in mvd, not in con3d')
for item in control_key:
    print(item, pretrained_weights_control[item].shape)



import json
import cv2
import numpy as np
import os , glob
from torch.utils.data import Dataset
import torch
from pathlib import Path
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler
import webdataset as wds
from omegaconf import DictConfig, ListConfig
import math
import matplotlib.pyplot as plt
import sys
from PIL import Image
import random
import argparse
import datetime
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from cldm.logger import ImageLogger
@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--finetune_from",
        type=str,
        nargs="?",
        default="",
        help="path to checkpoint to load model state from"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="resolution of image",
    )
    return parser

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view,  num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view
        image_transforms = [torchvision.transforms.Resize(256)]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        # total_view = 4
        # print("t1 train_data")
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                             sampler=sampler)

    def val_dataloader(self):
        # print("t1 val_data")
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        # print("t1 test_data")
        return wds.WebLoader(
            ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation), \
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.mean(image)
    # apply automatic Canny edge detection using the computed media

    lower = int(max(10, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    print(v, lower, upper)
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def random_canny(image):
    # compute the median of the single channel pixel intensities
    # apply automatic Canny edge detection using the computed media

    lower = 10 + np.random.random() * 90 # lower 0 ~ 100
    upper = 150 + np.random.random()*100 # upper 150 ~ 250

    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


class ObjaverseData(Dataset):
    def __init__(self,
                 root_dir='.objaverse/hf-objaverse-v1/views',
                 image_transforms=[],
                 ext="png",
                 default_trans=torch.zeros(3),
                 postprocess=None,
                 return_paths=False,
                 total_view=12,
                 validation=False
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths

        # if isinstance(postprocess, DictConfig):
        #     postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        # if not isinstance(ext, (tuple, list, ListConfig)):
        #     ext = [ext]

        # with open(os.path.join(root_dir, 'test_paths.json')) as f:
        #     self.paths = json.load(f)

        with open('valid_paths.json') as f:
            self.paths = json.load(f)

        total_objects = len(self.paths)

        print("*********number of total objects", total_objects)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):]  # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]  # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2)  # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)

        # color = [1., 1., 1., 1.]

        try:
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            # read prompt from BLIP
            f = open(os.path.join(filename, "BLIP_best_text.txt") , 'r')
            prompt = f.readline()
            # get cond_im and target_im
            cond_im = cv2.imread(os.path.join(filename, '%03d.png' % index_cond))
            target_im = cv2.imread(os.path.join(filename, '%03d.png' % index_target))
            # test_im = cv2.imread("test.png")

            # print("*** cond_im.shape", cond_im.shape)
            # print("*** target_im.shape",target_im.shape)
            # print("*** test_im.shape",test_im.shape)

            # BGR TO RGB
            cond_im  = cv2.cvtColor(cond_im , cv2.COLOR_BGR2RGB)
            target_im  = cv2.cvtColor(target_im , cv2.COLOR_BGR2RGB)
            # get canny edge
            canny_r = random_canny(cond_im)
            # print("*** canny_r.shape", canny_r.shape)
            canny_r = canny_r[:,:,None]
            canny_r = np.concatenate([canny_r, canny_r, canny_r], axis=2)
            # print("*** canny_r.shape after concatenate", canny_r.shape)
            # normalize
            canny_r = canny_r.astype(np.float32) / 255.0
            target_im  = (target_im .astype(np.float32) / 127.5) - 1.0


        except:
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1')  # this one we know is valid
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            # read prompt from BLIP
            f = open(os.path.join(filename, "BLIP_best_text.txt") , 'r')
            prompt = f.readline()
            # get cond_im and target_im
            cond_im = cv2.imread(os.path.join(filename, '%03d.png' % index_cond))
            target_im = cv2.imread(os.path.join(filename, '%03d.png' % index_target))
            # BGR TO RGB
            cond_im  = cv2.cvtColor(cond_im , cv2.COLOR_BGR2RGB)
            target_im  = cv2.cvtColor(target_im , cv2.COLOR_BGR2RGB)
            # get canny edge
            canny_r = random_canny(cond_im)
            # normalize
            canny_r = canny_r.astype(np.float32) / 255.0
            target_im  = (target_im .astype(np.float32) / 127.5) - 1.0



        data["img"] = target_im
        data["hint"] = canny_r
        data["camera_pose"] = self.get_T(target_RT, cond_RT) # actually the difference between two camera
        data["txt"] = prompt

        # print("test prompt is ", prompt)
        # print("img shape", target_im.shape, "hint shape", canny_r.shape)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))
#
#
#
# # setting for training
# batch_size= 20
# gpus=1
# # total batch = batch_size * gpus
# root_dir = '/yuch_ws/views_release'
# # setting for local test
# # batch_size= 1
# # gpus=1
# # root_dir = 'objvarse_views'
#
# num_workers = 16
# total_view = 12
# logger_freq = 300
# learning_rate = 1e-5
# sd_locked = True
# only_mid_control = False
#
#
# dataset = ObjaverseDataModuleFromConfig(root_dir, batch_size, total_view,  num_workers)
# dataset.prepare_data()
# dataset.setup()
#
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# # from tutorial_dataset import MyDataset
# from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
# #
# #
# # Configs
# # resume_path0 = './models/control_sd15_ini.ckpt'  # totorial
# # resume_path1 = './models/control_v11p_sd15_canny.pth'  # conv 1.1
# resume_path = 'models/control_sd15_canny.pth'  # conv 1
#
#
# # # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
# model = create_model('models/yuch_v11p_sd15_canny.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
# # model.load_state_dict(torch.load(resume_path))
# model.learning_rate = learning_rate
# model.sd_locked = sd_locked
# model.only_mid_control = only_mid_control
# #
# #
# # # Misc
# #
# #dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
# logger = ImageLogger(batch_frequency=logger_freq)
# # trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
# trainer = pl.Trainer(accelerator="gpu", devices=gpus, strategy="ddp", precision=32, callbacks=[logger])
#
# # Train!
# trainer.fit(model, dataset)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    print("***opt is ", opt)
    print("***unkown is ", unknown)

    if opt.resume:

        # resume training
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])  # eg : logs/2023_08_01_training
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        # first time training
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    imgdir = os.path.join(logdir, "images")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        # print("**configs", configs)

        cli = OmegaConf.from_dotlist(unknown)
        # print("**cli", cli) # cli should be NOne

        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())


        # print("***lc", lightning_config)
        # {'find_unused_parameters': False, 'metrics_over_trainsteps_checkpoint': True, 'modelcheckpoint':
        # {'params': {'every_n_train_steps': 500}}, 'callbacks': {'image_logger': {'target': 'main.ImageLogger',
        # 'params': {'batch_frequency': 500, 'max_images': 32, 'increase_log_steps': False, 'log_first_step': True,
        # 'log_images_kwargs': {'use_ema_scope': False, 'inpaint': False, 'plot_progressive_rows': False,
        # 'plot_diffusion_rows': False, 'N': 32, 'unconditional_guidance_scale': 3.0, 'unconditional_guidance_label':
        # ['']}}}}, 'trainer': {'benchmark': True, 'val_check_interval': 5000000, 'num_sanity_val_steps': 0,
        # 'accumulate_grad_batches': 1}}

        # print("***config2", config)
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # trainer:
        # benchmark: True
        # val_check_interval: 5000000  # really sorry
        # num_sanity_val_steps: 0
        # accumulate_grad_batches: 1

        # default to ddp
        trainer_config["accelerator"] = "ddp"

        print("*** nondefault_trainer_args:", nondefault_trainer_args(opt))
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            rank_zero_print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config
        print("*** config.model is ", config.model)
        # model
        print("model path = ", opt.base[0])
        model = create_model(opt.base[0]).cpu()
        model.cpu()
        print("***model load is done")

        print("\n\n***config model : " ,config.model.base_learning_rate, config.model.sd_locked,config.model.only_mid_control)
        model.learning_rate = config.model.base_learning_rate
        model.sd_locked = config.model.sd_locked
        model.only_mid_control = config.model.only_mid_control

        # # Configs
        # # resume_path0 = './models/control_sd15_ini.ckpt'  # totorial
        # # resume_path1 = './models/control_v11p_sd15_canny.pth'  # conv 1.1
        # resume_path = 'models/control_sd15_canny.pth'  # conv 1
        # # # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        # model = create_model('models/yuch_v11p_sd15_canny.yaml').cpu()
        # model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
        # # model.load_state_dict(torch.load(resume_path))
        # model.learning_rate = learning_rate
        # model.sd_locked = sd_locked
        # model.only_mid_control = only_mid_control
        # #dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
        # logger = ImageLogger(batch_frequency=logger_freq)
        # # trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
        # trainer = pl.Trainer(accelerator="gpu", devices=gpus, strategy="ddp", precision=32, callbacks=[logger])
        # # Train!
        # trainer.fit(model, dataset)


        if not opt.finetune_from == "":
            # we are finetuning from a ckpt
            rank_zero_print(f"Attempting to load state from {opt.finetune_from}")



            # old_state = torch.load(opt.finetune_from, map_location="cpu")

            # if "state_dict" in old_state:
            #     rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
            #     old_state = old_state["state_dict"]
            #
            # # Check if we need to port weights from 4ch input to 8ch
            # in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
            # new_state = model.state_dict()
            # in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
            # in_shape = in_filters_current.shape
            # if in_shape != in_filters_load.shape:
            #     input_keys = [
            #         "model.diffusion_model.input_blocks.0.0.weight",
            #         "model_ema.diffusion_modelinput_blocks00weight",
            #     ]
            #
            #     for input_key in input_keys:
            #         if input_key not in old_state or input_key not in new_state:
            #             continue
            #         input_weight = new_state[input_key]
            #         if input_weight.size() != old_state[input_key].size():
            #             print(f"Manual init: {input_key}")
            #             input_weight.zero_()
            #             input_weight[:, :4, :, :].copy_(old_state[input_key])
            #             old_state[input_key] = torch.nn.parameter.Parameter(input_weight)

            m, u = model.load_state_dict(load_state_dict(opt.finetune_from, location='cpu'), strict=False)
            # m, u = model.load_state_dict(old_state, strict=False)

            print("missing parameters: " , len(m) , "unkown parameters: ", len(u))
            if len(m) > 0:
                rank_zero_print("missing keys:")
                rank_zero_print(m)
            if len(u) > 0:
                rank_zero_print("unexpected keys:")
                rank_zero_print(u)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            rank_zero_print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        rank_zero_print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        # if version.parse(pl.__version__) < version.parse('1.4.0'):
        #     trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        # if version.parse(pl.__version__) >= version.parse('1.4.0'):
        #     default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            rank_zero_print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': 3,
                         'every_n_train_steps': 10,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        print("*** callbacks_cfg", callbacks_cfg)
        # callbacks_cfg["checkpoint_callback"]["params"]['save_top_k'] = -1
        # # callbacks_cfg["checkpoint_callback"]["params"]['save_last'] = None
        # callbacks_cfg["checkpoint_callback"]["params"]['filename'] = '{epoch}-{step}'
        # # callbacks_cfg["checkpoint_callback"]["params"]['mode'] = 'min'
        # callbacks_cfg["checkpoint_callback"]["params"]['monitor'] = 'global_step'
        # del callbacks_cfg["checkpoint_callback"]["params"]['save_top_k']
        # del callbacks_cfg["checkpoint_callback"]["params"]['save_last']
        # del callbacks_cfg["checkpoint_callback"]["params"]['every_n_train_steps']
        # print("**** callbacks_cfg", callbacks_cfg)
        # from datetime import timedelta
        # delta = timedelta(
        #     minutes=1,
        # )

        # val/loss_simple_ema
        # trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # personalization:
        # trainer_kwargs["callbacks"][-1].CHECKPOINT_NAME_LAST = "{epoch}-{step}--last"

        print("plugins in trainer_kwargs? " , "plugins" in trainer_kwargs)
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()

        # print("not lightning_config.get : ", not lightning_config.get("find_unused_parameters", True))
        if not lightning_config.get("find_unused_parameters", True):
            print("not lightning_config.get : ", not lightning_config.get("find_unused_parameters", True))
            from pytorch_lightning.plugins import DDPPlugin

            trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=True))

        from pytorch_lightning.plugins import DDPPlugin
        # save ckpt every n steps:
        # checkpoint_callback2 = ModelCheckpoint( monitor='global_step',save_last=True,filename='*cb2{epoch}-{step}', every_n_train_steps=5)
        # trainer_kwargs["callbacks"].append(checkpoint_callback2)

        logger = ImageLogger(batch_frequency=300, log_dir=imgdir)

        from pytorch_lightning.callbacks import ModelCheckpoint

        # print("***ckpt dir is :" , ckptdir)
        checkpoint_callback = ModelCheckpoint(monitor = 'global_step',dirpath = ckptdir,
                                              filename = 'control_{epoch}-{step}',verbose=True,
                                              every_n_train_steps=500, save_top_k=-1, save_last=True)


        trainer_kwargs["callbacks"] = [logger, checkpoint_callback]
        print("*** trainer opt " , trainer_opt)
        print("*** trainer kwargs " , trainer_kwargs)
        # gpus = '0,'
        # gpus = '0,1,2,3,4,5,6,7'
        trainer = pl.Trainer(accelerator="ddp", gpus = '0,1,2,3,4,5,6,7', precision=32, callbacks=[logger, checkpoint_callback])
        # trainer = Trainer.from_argparse_args(trainer_opt)
        print("*** log dir is " , logdir)
        trainer.logdir = logdir  ###
        # trainer = Trainer(plugins=[DDPPlugin(find_unused_parameters=False)] , accelerator='ddp',
        #                   accumulate_grad_batches=1, benchmark=True, gpus='0,', num_sanity_val_steps=0, val_check_interval=5000000 )
        # # setting for training
        batch_size = 20
        root_dir = '/yuch_ws/views_release'
        num_workers = 16
        total_view = 12
        logger_freq = 300

        data = ObjaverseDataModuleFromConfig(root_dir, batch_size, total_view, num_workers)
        data.prepare_data()
        data.setup()

        # data = instantiate_from_config(config.data)
        # data.prepare_data()
        # data.setup()
        rank_zero_print("#### Data ####")
        try:
            for k in data.datasets:
                rank_zero_print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        except:
            rank_zero_print("datasets not yet initialized.")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
            # ngpu = lightning_config.trainer.gpus
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            rank_zero_print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            rank_zero_print("++++ NOT USING LR SCALING ++++")
            rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                rank_zero_print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                if not opt.debug:
                    melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except RuntimeError as err:
        raise err
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            rank_zero_print(trainer.profiler.summary())








